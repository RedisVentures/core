// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <iterator>

#include "redis_cache.h"
#include "infer_stats.h"
#include "triton/common/logging.h"
#include "redis.cc"

namespace {

enum class ScopedTimerType { INSERTION, LOOKUP };

class ScopedTimer {
 public:
  explicit ScopedTimer(
      triton::core::InferenceRequest& request, uint64_t& duration,
      ScopedTimerType type)
      : request_(request), duration_(duration), type_(type)
  {
    switch (type_) {
      case ScopedTimerType::LOOKUP:
        request_.CaptureCacheLookupStartNs();
        break;
      case ScopedTimerType::INSERTION:
        request_.CaptureCacheInsertionStartNs();
        break;
    }
  }

  ~ScopedTimer()
  {
    switch (type_) {
      case ScopedTimerType::LOOKUP:
        request_.CaptureCacheLookupEndNs();
        duration_ +=
            request_.CacheLookupEndNs() - request_.CacheLookupStartNs();
        break;
      case ScopedTimerType::INSERTION:
        request_.CaptureCacheInsertionEndNs();
        duration_ +=
            request_.CacheInsertionEndNs() - request_.CacheInsertionStartNs();
        break;
    }
  }

 private:
  triton::core::InferenceRequest& request_;
  uint64_t& duration_;
  ScopedTimerType type_;
};

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

namespace triton { namespace core {

Status
RequestResponseCache::Create(
    std::string address,
    std::string username,
    std::string password,
    std::unique_ptr<RequestResponseCache>* cache)
{
  try {
    cache = &std::unique_ptr<RequestResponseCache>(
      new RequestResponseCache(address, username, password));
  }
  catch (const std::exception& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to initialize Response Cache: " + std::string(ex.what()));
  }

  return Status::Success;
}

RequestResponseCache::RequestResponseCache(std::string address, std::string username, std::string password)
{

  try {
    this->_client = init_client(address, username, password);
  }
  catch (const std::exception& ex) {
    throw std::runtime_error(
        "Failed to initialize Redis Response Cache: " + std::string(ex.what()));
  }

  LOG_INFO << "Redis Response Cache is located at:" << address;
}

RequestResponseCache::~RequestResponseCache()
{
  this->_client.reset();
}

bool RequestResponseCache::Exists(const uint64_t key) {
  std::string string_key = std::to_string(key);
  return this->_client->exists(string_key);
}

Status
RequestResponseCache::Lookup(
    InferenceResponse* const response, InferenceRequest* const request)
{

  if (request == nullptr) {
    return Status(
        Status::Code::INTERNAL, "Cache Lookup passed a nullptr request");
  }

  // Capture start latency now and end latency when timer goes out of scope
  ScopedTimer timer(
      *request, total_lookup_latency_ns_, ScopedTimerType::LOOKUP);

  // Hash the request and set cache key if it hasn't already been set
  if (!request->CacheKeyIsSet()) {
    RETURN_IF_ERROR(HashAndSet(request));
  }
  uint64_t key = request->CacheKey();

  num_lookups_++;
  LOG_VERBOSE(1) << request->LogRequest()
                 << "Looking up key [" + std::to_string(key) + "] in cache.";

  // Search cache for request hash key
  bool found = this->Exists(key);

  if (!found) {
    num_misses_++;
    LOG_VERBOSE(1) << request->LogRequest()
                   << "MISS for key [" + std::to_string(key) + "] in cache.";
    return Status(
        Status::Code::INTERNAL,
        request->LogRequest() + "key not found in cache");
  }
  // clean this up
  std::string str_key = std::to_string(key);
  std::string_view str_key_view = std::string_view(str_key);
  auto entry = cache_get(this->_client, str_key_view);

  // If find succeeds, it's a cache hit
  num_hits_++;
  LOG_VERBOSE(1) << request->LogRequest()
                 << "HIT for key [" + std::to_string(key) + "] in cache.";

  // Build InferenceResponse from CacheEntry
  RETURN_IF_ERROR(BuildInferenceResponse(entry, response));

  return Status::Success;
}

Status
RequestResponseCache::Insert(
    const InferenceResponse& response, InferenceRequest* const request)
{
  // Lock on cache insertion
  //std::lock_guard<std::recursive_mutex> lk(cache_mtx_);

  if (request == nullptr) {
    return Status(
        Status::Code::INTERNAL, "Cache Insert passed a nullptr request");
  }

  // Capture start latency now and end latency when timer goes out of scope
  ScopedTimer timer(
      *request, total_insertion_latency_ns_, ScopedTimerType::INSERTION);

  // Hash the request and set cache key if it hasn't already been set
  if (!request->CacheKeyIsSet()) {
    RETURN_IF_ERROR(HashAndSet(request));
  }
  const uint64_t key = request->CacheKey();

  // Exit early if key already exists in cache
  // Search cache for request hash key
  bool found = this->Exists(key);

  if (found) {
    return Status(
        Status::Code::ALREADY_EXISTS, request->LogRequest() + "key [" +
                                          std::to_string(key) +
                                          "] already exists in cache");
  }

  // Construct cache entry from response
  auto cache_entry = RedisCacheEntry<std::string_view>();

  RETURN_IF_ERROR(BuildCacheEntry(response, &cache_entry));

  // Insert entry into cache
  LOG_VERBOSE(1) << request->LogRequest()
                 << "Inserting key [" + std::to_string(key) + "] into cache.";


  return Status::Success;
}



Status
RequestResponseCache::BuildCacheEntry(
    const InferenceResponse& response, RedisCacheEntry<std::string_view>* cache_entry)
{

  // Build cache entry data from response outputs
  int num_entries = 0;
  for (const auto& response_output : response.Outputs()) {
    HashEntry<std::string_view> cache_input;

    // Fetch output buffer details
    const void* response_buffer = nullptr;
    size_t response_byte_size = 0;
    TRITONSERVER_MemoryType response_memory_type;
    int64_t response_memory_type_id;
    void* userp;
    RETURN_IF_ERROR(response_output.DataBuffer(
        &response_buffer, &response_byte_size, &response_memory_type,
        &response_memory_type_id, &userp));

    // TODO: Handle other memory types
    if (response_memory_type != TRITONSERVER_MEMORY_CPU &&
        response_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    // Exit early if response buffer from output is invalid
    if (response_buffer == nullptr) {
      return Status(
          Status::Code::INTERNAL, "Response buffer from output was nullptr");
    }
    char* buffer;
    // Copy data from response buffer to cache entry output buffer
    // TODO: Handle other memory types
    std::memcpy(buffer, response_buffer, response_byte_size);

    cache_input.fields["buffer"] = std::string_view(buffer);
    // Set output metadata
    cache_input.fields["name"] = std::string_view(response_output.Name());
    cache_input.fields["datatype"] = std::string_view(triton::common::DataTypeToProtocolString(response_output.DType()));
    cache_input.fields["shape"] =  std::string_view(triton::common::DimsListToString(response_output.Shape()));
    cache_input.fields["size"] = std::string_view(std::to_string(response_byte_size));

    // Add each output to cache entry
    cache_entry->outputs.emplace_back(cache_input);
    num_entries += 1;
  }
  cache_entry->num_entries = num_entries;

  return Status::Success;
}

std::vector<int64_t> get_dims(std::string dim_str) {

    int start = dim_str.find_first_of("[");
    int end = dim_str.find_last_of("]");
    std::string dim_sub_str = dim_str.substr(start, end);

    std::stringstream test(dim_sub_str);
    std::string segment;
    std::vector<int64_t> dim_list;
    while(std::getline(test, segment, ','))
    {
      dim_list.push_back(static_cast<int64_t>(segment.data(), sizeof(int64_t)));
    }
    return dim_list;
}

Status
RequestResponseCache::BuildInferenceResponse(
    const RedisCacheEntry<std::string>& entry, InferenceResponse* response)
{
  if (response == nullptr) {
    return Status(Status::Code::INTERNAL, "invalid response ptr passed in");
  }


  // Inference response outputs should be empty so we can append to them
  if (response->Outputs().size() != 0) {
    return Status(
        Status::Code::INTERNAL,
        "InferenceResponse already contains some outputs");
  }

  for (HashEntry<std::string> output : entry.outputs) {

    std::string dim_str = output.fields["shape"];
    std::vector<int64_t> dims = get_dims(dim_str);

    InferenceResponse::Output* response_output = nullptr;
    RETURN_IF_ERROR(response->AddOutput(
        output.fields["name"],
        triton::common::ProtocolStringToDataType(output.fields["datatype"]),
        dims,
        &response_output));

    if (response_output == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "InferenceResponse::Output pointer as nullptr");
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    // Allocate buffer for inference response
    void* buffer;
    size_t buffer_size = static_cast<size_t>(output.fields["size"].data(), sizeof(size_t));
    RETURN_IF_ERROR(response_output->AllocateDataBuffer(
        &buffer, buffer_size, &memory_type, &memory_type_id));

    // TODO: Handle other memory types
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    if (buffer == nullptr) {
      return Status(
          Status::Code::INTERNAL, "failed to allocate buffer for output '" +
                                      output.fields["name"] + "'");
    }
    // Copy cached output buffer to allocated response output buffer
    std::memcpy(buffer, output.fields["buffer"].data(), buffer_size);

    // TODO: Add field to InferenceResponse to indicate this was from cache
    // response.cached = true;
  }
  return Status::Success;
}

Status
RequestResponseCache::HashInputBuffers(
    const InferenceRequest::Input* input, size_t* seed)
{
  // Iterate over each data buffer in input in case of non-contiguous memory
  for (size_t idx = 0; idx < input->DataBufferCount(); ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RETURN_IF_ERROR(input->DataBuffer(
        idx, &src_buffer, &src_byte_size, &src_memory_type,
        &src_memory_type_id));

    // TODO: Handle other memory types
    if (src_memory_type != TRITONSERVER_MEMORY_CPU &&
        src_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    // Add each byte of input buffer chunk to hash
    const unsigned char* tmp = static_cast<const unsigned char*>(src_buffer);
    for (uint64_t byte = 0; byte < src_byte_size; byte++) {
      boost::hash_combine(*seed, tmp[byte]);
    }
  }

  return Status::Success;
}


/*

 tatus RequestReponseCache::SetJson(triton::common::TritonJson::Value& cache_entry) {

  triton::common::TritonJson::WriteBuffer* buf = triton::common::TritonJson::WriteBuffer();
  cache_entry->Write(buf);
  std::string* contents = buf->Contents();

  try {
    this->_client.command<void>("JSON.SET", "doc", ".", &contents);
  }
  catch (sw::redis::TimeoutError &e) {
      LOG_ERROR << request->LogRequest() << "Failed to insert key into cache.";
      return Status(
          Status::Code::INTERNAL,
          request->LogRequest() + "Cache insertion failed");
  }
  catch (sw::redis::IoError &e) {
      LOG_ERROR << request->LogRequest() << "Failed to insert key into cache.";
    return Status(
        Status::Code::INTERNAL,
        request->LogRequest() + "Cache insertion failed");
    }
  catch (...) {
    LOG_ERROR << request->LogRequest() << "Failed to insert key into cache.";
    return Status(
        Status::Code::INTERNAL,
        request->LogRequest() + "Cache insertion failed");
  }
  return Status::Success
}
 */



Status
RequestResponseCache::HashInputs(const InferenceRequest& request, size_t* seed)
{
  const auto& inputs = request.ImmutableInputs();
  // Convert inputs to ordered map for consistency in hashing
  // inputs sorted by key (input) name
  std::map<std::string, InferenceRequest::Input*> ordered_inputs(
      inputs.begin(), inputs.end());
  for (const auto& input : ordered_inputs) {
    // Add input name to hash
    boost::hash_combine(*seed, input.second->Name());
    // Fetch input buffer for hashing raw data
    RETURN_IF_ERROR(HashInputBuffers(input.second, seed));
  }

  return Status::Success;
}


Status
RequestResponseCache::Hash(const InferenceRequest& request, uint64_t* key)
{
  std::size_t seed = 0;
  // Add request model name to hash
  boost::hash_combine(seed, request.ModelName());
  // Add request model version to hash
  boost::hash_combine(seed, request.ActualModelVersion());
  RETURN_IF_ERROR(HashInputs(request, &seed));
  *key = static_cast<uint64_t>(seed);
  return Status::Success;
}

Status
RequestResponseCache::HashAndSet(InferenceRequest* const request)
{
  uint64_t key = 0;
  RETURN_IF_ERROR(Hash(*request, &key));
  request->SetCacheKey(key);
  return Status::Success;
}

}}  // namespace triton::core
