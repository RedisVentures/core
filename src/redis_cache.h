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
#include <unordered_map>

#include "infer_request.h"
#include "infer_response.h"
#include "model.h"
#include "status.h"

#include <sw/redis++/redis++.h>
#include "triton/common/triton_json.h"

#include <boost/functional/hash.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>


template <typename Tkv>
struct HashEntry {
  std::unordered_map<Tkv, Tkv> fields;
};

template <typename Tkv>
struct RedisCacheEntry {
  std::string_view key;
  int num_entries = 1;
  std::vector<HashEntry<Tkv>> outputs;
};


namespace triton { namespace core {


class RequestResponseCache {
 public:
  ~RequestResponseCache();
  // Create the request/response cache object
  static Status Create(std::string address, std::string username, std::string password, std::unique_ptr<RequestResponseCache>* cache);

  // Hash inference request for cache access and store it in "request" object.
  // This will also be called internally in Lookup/Insert if the request hasn't
  // already stored it's hash. It is up to the user to update the hash in the
  // request if modifying any hashed fields of the request object after storing.
  // Return Status object indicating success or failure.
  Status HashAndSet(InferenceRequest* const request);

  // Lookup 'request' hash in cache and return the inference response in
  // 'response' on cache hit or nullptr on cache miss
  // Return Status object indicating success or failure.
  Status Lookup(
      InferenceResponse* const response, InferenceRequest* const request);
  // Insert response into cache, evict entries to make space if necessary
  // Return Status object indicating success or failure.
  Status Insert(
      const InferenceResponse& response, InferenceRequest* const request);

  bool Exists(const uint64_t key);

  // Returns number of items in cache
  size_t NumEntries()
  {
    return 0; // TODO fix later
  }
  // Returns number of items evicted in cache lifespan
  size_t NumEvictions()
  {
    return num_evictions_;
  }
  // Returns number of lookups in cache lifespan, should sum to hits + misses
  size_t NumLookups()
  {
    return num_lookups_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumHits()
  {
    return num_hits_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumMisses()
  {
    return num_misses_;
  }
  // Returns the total lookup latency (nanoseconds) of all lookups in cache
  // lifespan
  uint64_t TotalLookupLatencyNs()
  {
    return total_lookup_latency_ns_;
  }

  uint64_t TotalInsertionLatencyNs()
  {
    return total_insertion_latency_ns_;
  }

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
  // Returns fraction of bytes allocated over total cache size between [0, 1]
  double TotalUtilization()
  {
    return 0; //TODO fix later
  }

 private:
  explicit RequestResponseCache(std::string address, std::string username, std::string password);

  // Build CacheEntry from InferenceResponse
  Status BuildCacheEntry(
      const InferenceResponse& response, RedisCacheEntry<std::string_view>* cache_entry);
  // Build InferenceResponse from CacheEntry
  Status BuildInferenceResponse(
      const RedisCacheEntry<std::string>& entry, InferenceResponse* response);
  // Helper function to hash data buffers used by "input"
  Status HashInputBuffers(const InferenceRequest::Input* input, size_t* seed);
  // Helper function to hash each input in "request"
  Status HashInputs(const InferenceRequest& request, size_t* seed);
  // Helper function to hash request and store it in "key"
  Status Hash(const InferenceRequest& request, uint64_t* key);

  std::unique_ptr<sw::redis::Redis> _client;
  // Cache metrics
  size_t num_evictions_ = 0;
  size_t num_lookups_ = 0;
  size_t num_hits_ = 0;
  size_t num_misses_ = 0;
  uint64_t total_lookup_latency_ns_ = 0;
  uint64_t total_insertion_latency_ns_ = 0;
};

}}  // namespace triton::core
