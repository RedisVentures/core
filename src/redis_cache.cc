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

#include <sstream>
#include <cstdio>
#include <iterator>

#include "redis_cache.h"


std::unique_ptr<sw::redis::Redis> init_client(
    const std::string& address,
    const std::string& user_name,
    const std::string& password) {
  // Put together cluster configuration.
  sw::redis::ConnectionOptions options;

  const std::string::size_type comma_pos = address.find(',');
  const std::string host = comma_pos == std::string::npos ? address : address.substr(0, comma_pos);
  const std::string::size_type colon_pos = host.find(':');
  if (colon_pos == std::string::npos) {
    options.host = host;
  } else {
    options.host = host.substr(0, colon_pos);
    options.port = std::stoi(host.substr(colon_pos + 1));
  }
  options.user = user_name;
  options.password = password;
  options.keep_alive = true;

  sw::redis::ConnectionPoolOptions pool_options;
  pool_options.size = 1;

  // Connect to cluster.
  std::cout << "Connecting via " << options.host << ':' << options.port << "..." << std::endl;
  std::unique_ptr<sw::redis::Redis> redis = std::make_unique<sw::redis::Redis>(options, pool_options);
  return redis;
}


namespace triton { namespace core {


Status
RedisResponseCache::Create(
    std::string address,
    std::string username,
    std::string password,
    std::unique_ptr<RequestResponseCache>* cache)
{
  try {
    cache->reset(new RedisResponseCache(address, username, password));
  }
  catch (const std::exception& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to initialize Response Cache: " + std::string(ex.what()));
  }

  return Status::Success;
}

RedisResponseCache::RedisResponseCache(std::string address, std::string username, std::string password)
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

RedisResponseCache::~RedisResponseCache()
{
  this->_client.reset();
}

bool RedisResponseCache::Exists(const uint64_t key) {
  std::string key_s = std::to_string(key);
  return this->_client->exists(key_s);
}

Status
RedisResponseCache::Lookup(
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
    RETURN_IF_ERROR(RequestResponseCache::HashAndSet(request));
  }
  uint64_t key = request->CacheKey();

  num_lookups_++;
  LOG_VERBOSE(1) << request->LogRequest()
                 << "Looking up key [" + std::to_string(key) + "] in cache.";

  // Search cache for request hash key
  bool found = this->Exists(key); //TODO fix this.

  if (!found) {
    num_misses_++;
    LOG_VERBOSE(1) << request->LogRequest()
                   << "MISS for key [" + std::to_string(key) + "] in cache.";
    return Status(
        Status::Code::INTERNAL,
        request->LogRequest() + "key not found in cache");
  }
  // clean this up
  const std::string& key_v = std::to_string(key);

  RedisCacheEntry<std::string> entry;
  entry.key = key_v;
  RETURN_IF_ERROR(cache_get(&entry));

  // If find succeeds, it's a cache hit
  num_hits_++;
  LOG_VERBOSE(1) << request->LogRequest()
                 << "HIT for key [" + std::to_string(key) + "] in cache.";

  // Build InferenceResponse from CacheEntry
  RETURN_IF_ERROR(BuildInferenceResponse(entry, response));

  return Status::Success;
}

Status
RedisResponseCache::Insert(
    const InferenceResponse& response, InferenceRequest* const request)
{

  if (request == nullptr) {
    return Status(
        Status::Code::INTERNAL, "Cache Insert passed a nullptr request");
  }

  // Capture start latency now and end latency when timer goes out of scope
  ScopedTimer timer(
      *request, total_insertion_latency_ns_, ScopedTimerType::INSERTION);

  // Hash the request and set cache key if it hasn't already been set
  if (!request->CacheKeyIsSet()) {
    RETURN_IF_ERROR(RequestResponseCache::HashAndSet(request));
  }
  uint64_t key = request->CacheKey();

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
  auto cache_entry = RedisCacheEntry<std::string>();
  cache_entry.key = std::to_string(key);

  RETURN_IF_ERROR(BuildCacheEntry(response, &cache_entry));
  RETURN_IF_ERROR(cache_set(cache_entry));

  // Insert entry into cache
  LOG_VERBOSE(1) << request->LogRequest()
                 << "Inserting key [" + std::to_string(key) + "] into cache.";


  return Status::Success;
}


Status
RedisResponseCache::BuildCacheEntry(
    const InferenceResponse& response, RedisCacheEntry<std::string>* cache_entry)
{

  // Build cache entry data from response outputs
  int num_entries = 0;
  for (const auto& response_output : response.Outputs()) {

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
    void* buffer = malloc(response_byte_size);
    // Copy data from response buffer to cache entry output buffer
    std::memcpy(buffer, response_buffer, response_byte_size);

    // convert datatypes
    const std::string& byte_size = std::to_string(response_byte_size);
    const std::string& buf_shape = triton::common::DimsListToString(response_output.Shape());
    const std::string& buf = std::string((char*)buffer, response_byte_size);
    const std::string& dt = triton::common::DataTypeToProtocolString(response_output.DType());

    cache_entry->fields.insert({
      {suffix_key("size", num_entries), byte_size},
      {suffix_key("dtype", num_entries), dt},
      {suffix_key("shape", num_entries), buf_shape},
      {suffix_key("buffer", num_entries), buf},
      {suffix_key("name", num_entries), response_output.Name()}
    });

    // Add each output to cache entry
    num_entries += 1;
  }
  cache_entry->num_entries = num_entries;

  return Status::Success;
}

std::string RedisResponseCache::suffix_key(std::string key, int suffix) {
  std::string s_key = key + "_" + std::to_string(suffix);
  return s_key;
}

std::vector<int64_t> RedisResponseCache::get_dims(std::string dim_str) {

    int start = dim_str.find_first_of("[");
    int end = dim_str.find_last_of("]");
    std::string dim_sub_str = dim_str.substr(start+1, end-1);

    std::stringstream test(dim_sub_str);
    std::string segment;
    std::vector<int64_t> dim_list;
    while(std::getline(test, segment, ','))
    {
      int64_t dim = std::stoi(segment);
      dim_list.push_back(dim);
    }
    return dim_list;
}

Status
RedisResponseCache::BuildInferenceResponse(
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

  for (int i = 0; i < entry.num_entries; i++) {

    // access fields in cache entry hash map
    const std::string& name = entry.fields.at(suffix_key("name", i));
    const std::string& dtype = entry.fields.at(suffix_key("dtype", i));
    const std::string& bufsize = entry.fields.at(suffix_key("size", i));
    const std::string& shape = entry.fields.at(suffix_key("shape", i));
    const std::string& buf = entry.fields.at(suffix_key("buffer", i));


    // convert datatypes back to InferenceResponse datatypes
    // get dimensions
    std::vector<int64_t> dims = get_dims(shape);
    // get datatype
    inference::DataType datatype = triton::common::ProtocolStringToDataType(dtype);
    //cast buffer size to size_t
    std::stringstream sstream(bufsize);
    size_t buffer_size;
    sstream >> buffer_size;

    InferenceResponse::Output* response_output = nullptr;
    RETURN_IF_ERROR(response->AddOutput(
        name,
        datatype,
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
                                      name + "'");
    }
    std::memcpy(buffer, buf.data(), buffer_size);

    // TODO: Add field to InferenceResponse to indicate this was from cache
    // response.cached = true;
  }
  return Status::Success;
}

Status
RedisResponseCache::cache_set(RedisCacheEntry<std::string> &cache_entry) {

    // set number of entries in the top level
    const std::string& entries_k = "entries";
    const std::string& entries_v = std::to_string(cache_entry.num_entries);
    cache_entry.fields.insert({entries_k, entries_v});

    // set response in a redis hash field
    try {
      _client->hmset(
        cache_entry.key,
        cache_entry.fields.begin(),
        cache_entry.fields.end()
      );
    }
    catch (sw::redis::TimeoutError &e) {
      std::string err = "Timeout inserting key" + cache_entry.key + " into cache.";
      LOG_ERROR << err << "\n Failed with error " + std::string(e.what());
      return Status(
          Status::Code::INTERNAL, err);
    }
    catch (sw::redis::IoError &e) {
      std::string err = "Failed to insert key" + cache_entry.key + " into cache.";
      LOG_ERROR << err << "\n Failed with error " + std::string(e.what());
      return Status(
          Status::Code::INTERNAL, err);
      }
    catch (...) {
      std::string err = "Failed to insert key" + cache_entry.key + " into cache.";
      LOG_ERROR << err;
      return Status(
          Status::Code::INTERNAL, err);
      }
    return Status::Success;
  }



Status
RedisResponseCache::cache_get(RedisCacheEntry<std::string> *cache_entry) {
    try {
      _client->hgetall(
        cache_entry->key,
        std::inserter(cache_entry->fields, cache_entry->fields.begin())
      );

    }
    catch (sw::redis::TimeoutError &e) {
      std::string err = "Timeout inserting key" + cache_entry->key + " into cache.";
      LOG_ERROR << err << "\n Failed with error " + std::string(e.what());
      return Status(
          Status::Code::INTERNAL, err);
    }
    catch (sw::redis::IoError &e) {
      std::string err = "Failed to insert key" + cache_entry->key + " into cache.";
      LOG_ERROR << err << "\n Failed with error " + std::string(e.what());
      return Status(
          Status::Code::INTERNAL, err);
    }
    catch (...) {
      std::string err = "Failed to insert key" + cache_entry->key + " into cache.";
      LOG_ERROR << err;
      return Status(
          Status::Code::INTERNAL, err);
    }

    // emptiness check
    if (cache_entry->fields.empty()) {
      return Status(
        Status::Code::INTERNAL,
        "Failed to retrieve key from remote cache");
    }

    // set number of entries at the top level
    const char* entries = cache_entry->fields.at("entries").c_str();
    cache_entry->num_entries = std::atoi(entries);

    return Status::Success;
  }

}}  // namespace triton::core
