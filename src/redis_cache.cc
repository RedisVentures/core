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


std::string prefix_key(const uint64_t key) {
  //use hash_tag here
  const std::string s_key = std::to_string(key);
  std::string prefix = "{" + s_key + "}";
  return prefix;
}


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

void cache_set(
  std::unique_ptr<sw::redis::Redis> &redis,
  RedisCacheEntry<std::string> &cache_entry) {

    //use hash_tag here
    const std::string& prefix = "{" + cache_entry.key + "}";

    // set number of entries in the top level
    const std::string& entries_v = std::to_string(cache_entry.num_entries);

    //set num entries prefix
    bool is_set = redis->set(prefix, entries_v);
    std::cout << std::to_string(is_set) << std::endl;


    int i = 1;
    for (auto &output: cache_entry.outputs) {
      const std::string& key_v = prefix + "." + std::to_string(i);

      // set response in a redis hash field
      redis->hmset(key_v, output.fields.begin(), output.fields.end());
      i += 1;
    }
}


RedisCacheEntry<std::string> cache_get(
    std::unique_ptr<sw::redis::Redis> &redis,
    const std::string& hash_key) {

    RedisCacheEntry<std::string> entry;

    //use hash_tag here
    const std::string& prefix = "{" + hash_key + "}";

    std::string num = redis->get(prefix).value();
    int num_entries = std::atoi(num.c_str());

    std::vector<HashEntry<std::string>> entries;

    /// TODO pipeline these
    for (int i=1; i <=num_entries; i++) {
      HashEntry<std::string> h;
      const std::string& key = prefix + "." + std::to_string(i);
      redis->hgetall(key, std::inserter(h.fields, h.fields.begin()));
      entries.emplace_back(h);
    }

    entry.key = prefix;
    entry.num_entries = num_entries;
    entry.outputs = entries;

    return entry;
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
  // flushall?
  this->_client.reset();
}

bool RedisResponseCache::Exists(const uint64_t key) {
  std::string prefixed_key = prefix_key(key);
  return this->_client->exists(prefixed_key);
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
  auto entry = cache_get(this->_client, key_v);

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
  //cache_entry.key = {reinterpret_cast<const char *>(&key), sizeof(uint64_t)};
  cache_entry.key = std::to_string(key);

  RETURN_IF_ERROR(BuildCacheEntry(response, &cache_entry));

  cache_set(this->_client, cache_entry);

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
    HashEntry<std::string> cache_input;

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

    const std::string& buf_str = "buffer";
    const std::string& name_str = "name";
    const std::string& dt_str = "datatype";
    const std::string& shape_str = "shape";
    const std::string& size_str = "size";

    const std::string& byte_size = std::to_string(response_byte_size);
    const std::string& buf_shape = triton::common::DimsListToString(response_output.Shape());
    const std::string& buf = std::string((char*)buffer, response_byte_size);

    const std::string& name = std::string(response_output.Name());
    const std::string& dt = triton::common::DataTypeToProtocolString(response_output.DType());

    cache_input.fields.insert({
      {buf_str, buf},
      {name_str, name},
      {dt_str, dt},
      {shape_str, buf_shape},
      {size_str, byte_size}
    });

      // for debugging
      //for (auto &field: cache_input.fields) {
      //  std::cout<< field.first << " = " << field.second << std::endl;
      //}

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

  for (HashEntry<std::string> output : entry.outputs) {


    const std::string& name = output.fields["name"];
    const std::string& dtype = output.fields["datatype"];
    const std::string& bufsize = output.fields["size"];
    const std::string& dim_str = output.fields["shape"];
    const std::string& buf = output.fields["buffer"];
    std::vector<int64_t> dims = get_dims(dim_str);

    // get datatype
    inference::DataType datatype = triton::common::ProtocolStringToDataType(dtype);

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

    //cast buffer size to size_t
    std::stringstream sstream(bufsize);
    size_t buffer_size;
    sstream >> buffer_size;

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

}}  // namespace triton::core
