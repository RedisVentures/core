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

#pragma once

#include <list>
#include <string>
#include <iostream>
#include <string_view>
#include <unordered_map>

#include "triton/common/model_config.h"
#include "triton/common/logging.h"
#include "infer_stats.h"
#include "response_cache.h"
#include <sw/redis++/redis++.h>


template <typename Tkv>
struct RedisCacheEntry {
  std::string key;
  int num_entries = 1;
  std::unordered_map<Tkv, Tkv> fields;
};


namespace triton { namespace core {


class RedisResponseCache : public RequestResponseCache {
 public:
  ~RedisResponseCache();

  // Create the request/response cache object
  static Status Create(std::string address, std::string username, std::string password, std::unique_ptr<RequestResponseCache>* cache);

  Status Lookup(
      InferenceResponse* const response, InferenceRequest* const request);

  Status Insert(
      const InferenceResponse& response, InferenceRequest* const request);

  bool Exists(const uint64_t key);

  // dummy method to statisfy compiler
  // there is bound to be a better way to do this.
  Status Evict() {
    Status status = Status(Status::Code::SUCCESS);
    return status;
  }

  Status Flush() {
    // empty the entire database (mostly used in testing)
    _client->flushall();
   return Status(Status::Code::SUCCESS);
  }

  // Returns number of items in cache
  size_t NumEntries()
  {
    // TODO think about if the cache is used for more than 1 purpose
    // this method counts keys.. scanning and keys commands are too slow
    // maybe just make this explicit?
    return (size_t)_client->dbsize();
  }
  // Returns number of items evicted in cache lifespan
  size_t NumEvictions()
  {
    // TODO get from Redis instead
    return RequestResponseCache::NumEvictions();
  }

  // TODO Are these needed for Redis? It's easy to get the values
  // Returns total number of bytes allocated for cache
  size_t TotalBytes()
  {
    return 0; //TODO fix later
  }
  // Returns number of free bytes in cache
  size_t FreeBytes()
  {
    return 0; //TODO fix later
  }
  // Returns number of bytes in use by cache
  size_t AllocatedBytes()
  {
    return 0; //TODO fix later
  }
  // Returns fraction of bytes allocated over total cache size
  double TotalUtilization()
  {
    return 0; //TODO fix later
  }

 private:
  explicit RedisResponseCache(std::string address, std::string username, std::string password);

  // Build CacheEntry from InferenceResponse
  Status BuildCacheEntry(
      const InferenceResponse& response, RedisCacheEntry<std::string>* cache_entry);
  // Build InferenceResponse from CacheEntry
  Status BuildInferenceResponse(
      const RedisCacheEntry<std::string>& entry, InferenceResponse* response);

  // get/set
  Status cache_set(RedisCacheEntry<std::string> &cache_entry);
  Status cache_get(RedisCacheEntry<std::string> *cache_entry);
  // helpers
  std::string suffix_key(std::string key, int suffix);
  std::vector<int64_t> get_dims(std::string dim_str);

  std::unique_ptr<sw::redis::Redis> _client;
};

}}  // namespace triton::core
