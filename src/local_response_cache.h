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
#include "response_cache.h"

#include <boost/interprocess/managed_external_buffer.hpp>

namespace triton { namespace core {

// Assuming CPU memory only for now
struct Output {
  // Output tensor data buffer
  void* buffer_;
  // Size of "buffer" above
  uint64_t buffer_size_ = 0;
  // Name of the output
  std::string name_;
  // Datatype of the output
  inference::DataType dtype_;
  // Shape of the output
  std::vector<int64_t> shape_;
};

struct CacheEntry {
  explicit CacheEntry() {}
  // Point to key in LRU list for maintaining LRU order
  std::list<uint64_t>::iterator lru_iter_;
  // each output buffer = managed_buffer.allocate(size, ...)
  std::vector<Output> outputs_;
};

class LocalResponseCache: public RequestResponseCache {
 public:
  ~LocalResponseCache();
  // Create the request/response cache object
  static Status Create(
      uint64_t cache_size, std::unique_ptr<RequestResponseCache>* cache);

  // Lookup 'request' hash in cache and return the inference response in
  // 'response' on cache hit or nullptr on cache miss
  // Return Status object indicating success or failure.
  Status Lookup(
      InferenceResponse* const response, InferenceRequest* const request);
  // Insert response into cache, evict entries to make space if necessary
  // Return Status object indicating success or failure.
  Status Insert(
      const InferenceResponse& response, InferenceRequest* const request);
  // Evict entry from cache based on policy
  // Return Status object indicating success or failure.
  Status Evict();

  // emtpy the cache
  Status Flush() {
    // TODO implement this.
    return Status(Status::Code::SUCCESS);
  }

  // Returns number of items in cache
  size_t NumEntries()
  {
    std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
    return cache_.size();
  }
  // Returns total number of bytes allocated for cache
  size_t TotalBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_size();
  }
  // Returns number of free bytes in cache
  size_t FreeBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_free_memory();
  }
  // Returns number of bytes in use by cache
  size_t AllocatedBytes()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return managed_buffer_.get_size() - managed_buffer_.get_free_memory();
  }
  // Returns fraction of bytes allocated over total cache size between [0, 1]
  double TotalUtilization()
  {
    std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
    return static_cast<double>(AllocatedBytes()) /
           static_cast<double>(TotalBytes());
  }

 private:
  explicit LocalResponseCache(const uint64_t cache_size);
  // Update LRU ordering on lookup
  void UpdateLRU(std::unordered_map<uint64_t, CacheEntry>::iterator&);
  // Build CacheEntry from InferenceResponse
  Status BuildCacheEntry(
      const InferenceResponse& response, CacheEntry* const entry);
  // Build InferenceResponse from CacheEntry
  Status BuildInferenceResponse(
      const CacheEntry& entry, InferenceResponse* const response);

  // Cache buffer
  void* buffer_;
  // Managed buffer
  boost::interprocess::managed_external_buffer managed_buffer_;
  // key -> CacheEntry containing values and list iterator for LRU management
  std::unordered_map<uint64_t, CacheEntry> cache_;
  // List of keys sorted from most to least recently used
  std::list<uint64_t> lru_;

  // Mutex for buffer synchronization
  std::recursive_mutex buffer_mtx_;
};

}}  // namespace triton::core
