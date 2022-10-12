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

#include "response_cache.h"
#include "infer_stats.h"
#include "triton/common/logging.h"


namespace triton { namespace core {

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