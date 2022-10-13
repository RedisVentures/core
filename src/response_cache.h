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

#include <string>
#include "infer_request.h"
#include "infer_response.h"
#include "status.h"
#include "model.h"

// for hashing input buffers/model name/model version
#include <boost/functional/hash.hpp>


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

}  // namespace


namespace triton { namespace core {


class RequestResponseCache {
  public:

  // Lookup 'request' hash in cache and return the inference response in
  // 'response' on cache hit or nullptr on cache miss
  // Return Status object indicating success or failure.
  virtual Status Lookup(
    InferenceResponse* const response, InferenceRequest* const request) = 0;

  virtual Status Insert(
    const InferenceResponse& response, InferenceRequest* const request) = 0;

  // overridden only in the LocalRequestCache class
  // TODO something else better here?
  virtual Status Evict() = 0;

  virtual Status Flush() = 0;

  // Hash inference request for cache access and store it in "request" object.
  // This will also be called internally in Lookup/Insert if the request hasn't
  // already stored it's hash. It is up to the user to update the hash in the
  // request if modifying any hashed fields of the request object after storing.
  // Return Status object indicating success or failure.
  Status HashAndSet(InferenceRequest* const request);

 // Returns number of items evicted in cache lifespan
  size_t NumEvictions()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_evictions_;
  }
  // Returns number of lookups in cache lifespan, should sum to hits + misses
  size_t NumLookups()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_lookups_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumHits()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_hits_;
  }
  // Returns number of cache hits in cache lifespan
  size_t NumMisses()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return num_misses_;
  }
  // Returns the total lookup latency (nanoseconds) of all lookups in cache
  // lifespan
  uint64_t TotalLookupLatencyNs()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return total_lookup_latency_ns_;
  }

  uint64_t TotalInsertionLatencyNs()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return total_insertion_latency_ns_;
  }
  // Returns number of items in cache
  virtual size_t NumEntries() = 0;
  // Returns total number of bytes allocated for cache
  virtual size_t TotalBytes() = 0;
  // Returns number of free bytes in cache
  virtual size_t FreeBytes() = 0;
  // Returns number of bytes in use by cache
  virtual size_t AllocatedBytes() = 0;
  // Returns fraction of bytes allocated over total cache size between [0, 1]
  virtual double TotalUtilization() = 0;

  protected:
  // Mutex for cache synchronization
  std::recursive_mutex cache_mtx_;
  // Cache metrics
  size_t num_evictions_ = 0;
  size_t num_lookups_ = 0;
  size_t num_hits_ = 0;
  size_t num_misses_ = 0;
  uint64_t total_lookup_latency_ns_ = 0;
  uint64_t total_insertion_latency_ns_ = 0;

  private:

  // Helper function to hash data buffers used by "input"
  Status HashInputBuffers(const InferenceRequest::Input* input, size_t* seed);
  // Helper function to hash each input in "request"
  Status HashInputs(const InferenceRequest& request, size_t* seed);
  // Helper function to hash request and store it in "key"
  Status Hash(const InferenceRequest& request, uint64_t* key);

};

}}