// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "perf_utils.h"

#include <fcntl.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "client_backend/client_backend.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {

cb::ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return cb::ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return cb::ProtocolType::GRPC;
  }
  return cb::ProtocolType::UNKNOWN;
}

cb::Error
ConvertDTypeFromTFS(const std::string& tf_dtype, std::string* datatype)
{
  if (tf_dtype == "DT_HALF") {
    *datatype = "FP16";
  } else if (tf_dtype == "DT_BFLOAT16") {
    *datatype = "BF16";
  } else if (tf_dtype == "DT_FLOAT") {
    *datatype = "FP32";
  } else if (tf_dtype == "DT_DOUBLE") {
    *datatype = "FP64";
  } else if (tf_dtype == "DT_INT32") {
    *datatype = "INT32";
  } else if (tf_dtype == "DT_INT16") {
    *datatype = "INT16";
  } else if (tf_dtype == "DT_UINT16") {
    *datatype = "UINT16";
  } else if (tf_dtype == "DT_INT8") {
    *datatype = "INT8";
  } else if (tf_dtype == "DT_UINT8") {
    *datatype = "UINT8";
  } else if (tf_dtype == "DT_STRING") {
    *datatype = "BYTES";
  } else if (tf_dtype == "DT_INT64") {
    *datatype = "INT64";
  } else if (tf_dtype == "DT_BOOL") {
    *datatype = "BOOL";
  } else if (tf_dtype == "DT_UINT32") {
    *datatype = "UINT32";
  } else if (tf_dtype == "DT_UINT64") {
    *datatype = "UINT64";
  } else {
    return cb::Error(
        "unsupported datatype encountered " + tf_dtype, pa::GENERIC_ERROR);
  }

  return cb::Error::Success;
}

bool
IsDirectory(const std::string& path)
{
  struct stat s;
  if (stat(path.c_str(), &s) == 0 && (s.st_mode & S_IFDIR)) {
    return true;
  } else {
    return false;
  }
}

bool
IsFile(const std::string& complete_path)
{
  struct stat s;
  if (stat(complete_path.c_str(), &s) == 0 && (s.st_mode & S_IFREG)) {
    return true;
  } else {
    return false;
  }
}

int64_t
ByteSize(const std::vector<int64_t>& shape, const std::string& datatype)
{
  int one_element_size;
  if ((datatype.compare("BOOL") == 0) || (datatype.compare("INT8") == 0) ||
      (datatype.compare("UINT8") == 0)) {
    one_element_size = 1;
  } else if (
      (datatype.compare("INT16") == 0) || (datatype.compare("UINT16") == 0) ||
      (datatype.compare("FP16") == 0) || (datatype.compare("BF16") == 0)) {
    one_element_size = 2;
  } else if (
      (datatype.compare("INT32") == 0) || (datatype.compare("UINT32") == 0) ||
      (datatype.compare("FP32") == 0)) {
    one_element_size = 4;
  } else if (
      (datatype.compare("INT64") == 0) || (datatype.compare("UINT64") == 0) ||
      (datatype.compare("FP64") == 0)) {
    one_element_size = 8;
  } else {
    return -1;
  }

  int64_t count = ElementCount(shape);
  if (count < 0) {
    return count;
  }

  return (one_element_size * count);
}

int64_t
ElementCount(const std::vector<int64_t>& shape)
{
  int64_t count = 1;
  bool is_dynamic = false;
  for (const auto dim : shape) {
    if (dim == -1) {
      is_dynamic = true;
    } else {
      count *= dim;
    }
  }

  if (is_dynamic) {
    count = -1;
  }
  return count;
}

void
SerializeStringTensor(
    std::vector<std::string> string_tensor, std::vector<char>* serialized_data)
{
  std::string serialized = "";
  for (auto s : string_tensor) {
    uint32_t len = s.size();
    serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    serialized.append(s);
  }

  std::copy(
      serialized.begin(), serialized.end(),
      std::back_inserter(*serialized_data));
}

cb::Error
SerializeExplicitTensor(
    const rapidjson::Value& tensor, const std::string& dt,
    std::vector<char>* decoded_data)
{
  if (dt.compare("BYTES") == 0) {
    std::string serialized = "";
    for (const auto& value : tensor.GetArray()) {
      if (!value.IsString()) {
        return cb::Error(
            "unable to find string data in json", pa::GENERIC_ERROR);
      }
      std::string element(value.GetString());
      uint32_t len = element.size();
      serialized.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
      serialized.append(element);
    }
    std::copy(
        serialized.begin(), serialized.end(),
        std::back_inserter(*decoded_data));
  } else if (dt.compare("JSON") == 0) {
    std::string serialized = "";

    auto values = tensor.GetArray();
    if (values.Size() != 1) {
      return cb::Error(
          "JSON format does not yet support multiple json objects in the "
          "input");
    }
    for (const auto& value : values) {
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      value.Accept(writer);

      std::string element = buffer.GetString();
      uint32_t len = element.size();
      serialized.append(element);
    }
    std::copy(
        serialized.begin(), serialized.end(),
        std::back_inserter(*decoded_data));
  } else {
    for (const auto& value : tensor.GetArray()) {
      if (dt.compare("BOOL") == 0) {
        if (!value.IsBool()) {
          return cb::Error(
              "unable to find bool data in json", pa::GENERIC_ERROR);
        }
        bool element(value.GetBool());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(bool));
      } else if (dt.compare("UINT8") == 0) {
        if (!value.IsUint()) {
          return cb::Error(
              "unable to find uint8_t data in json", pa::GENERIC_ERROR);
        }
        uint8_t element(static_cast<uint8_t>(value.GetUint()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint8_t));
      } else if (dt.compare("INT8") == 0) {
        if (!value.IsInt()) {
          return cb::Error(
              "unable to find int8_t data in json", pa::GENERIC_ERROR);
        }
        int8_t element(static_cast<int8_t>(value.GetInt()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int8_t));
      } else if (dt.compare("UINT16") == 0) {
        if (!value.IsUint()) {
          return cb::Error(
              "unable to find uint16_t data in json", pa::GENERIC_ERROR);
        }
        uint16_t element(static_cast<uint16_t>(value.GetUint()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint16_t));
      } else if (dt.compare("INT16") == 0) {
        if (!value.IsInt()) {
          return cb::Error(
              "unable to find int16_t data in json", pa::GENERIC_ERROR);
        }
        int16_t element(static_cast<int16_t>(value.GetInt()));
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int16_t));
      } else if (dt.compare("FP16") == 0) {
        return cb::Error(
            "Can not use explicit tensor description for fp16 datatype",
            pa::GENERIC_ERROR);
      } else if (dt.compare("BF16") == 0) {
        return cb::Error(
            "Can not use explicit tensor description for bf16 datatype",
            pa::GENERIC_ERROR);
      } else if (dt.compare("UINT32") == 0) {
        if (!value.IsUint()) {
          return cb::Error(
              "unable to find uint32_t data in json", pa::GENERIC_ERROR);
        }
        uint32_t element(value.GetUint());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint32_t));
      } else if (dt.compare("INT32") == 0) {
        if (!value.IsInt()) {
          return cb::Error(
              "unable to find int32_t data in json", pa::GENERIC_ERROR);
        }
        int32_t element(value.GetInt());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int32_t));
      } else if (dt.compare("FP32") == 0) {
        if (!value.IsDouble()) {
          return cb::Error(
              "unable to find float data in json", pa::GENERIC_ERROR);
        }
        float element(value.GetFloat());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(float));
      } else if (dt.compare("UINT64") == 0) {
        if (!value.IsUint64()) {
          return cb::Error(
              "unable to find uint64_t data in json", pa::GENERIC_ERROR);
        }
        uint64_t element(value.GetUint64());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(uint64_t));
      } else if (dt.compare("INT64") == 0) {
        if (!value.IsInt64()) {
          return cb::Error(
              "unable to find int64_t data in json", pa::GENERIC_ERROR);
        }
        int64_t element(value.GetInt64());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(int64_t));
      } else if (dt.compare("FP64") == 0) {
        if (!value.IsDouble()) {
          return cb::Error(
              "unable to find fp64 data in json", pa::GENERIC_ERROR);
        }
        double element(value.GetDouble());
        const char* src = reinterpret_cast<const char*>(&element);
        decoded_data->insert(decoded_data->end(), src, src + sizeof(double));
      } else {
        return cb::Error("Unexpected type " + dt);
      }
    }
  }
  return cb::Error::Success;
}

std::string
GetRandomString(const int string_length)
{
  std::mt19937_64 gen{std::random_device()()};
  std::uniform_int_distribution<size_t> dist{0, character_set.length() - 1};
  std::string random_string;
  std::generate_n(std::back_inserter(random_string), string_length, [&] {
    return character_set[dist(gen)];
  });
  return random_string;
}

std::string
ShapeVecToString(const std::vector<int64_t> shape_vec, bool skip_first)
{
  bool first = true;
  std::string str("[");
  for (const auto& value : shape_vec) {
    if (skip_first) {
      skip_first = false;
      continue;
    }
    if (!first) {
      str += ",";
    }
    str += std::to_string(value);
    first = false;
  }

  str += "]";
  return str;
}

std::string
TensorToRegionName(std::string name)
{
  // Remove slashes from the name, if any.
  name.erase(
      std::remove_if(
          name.begin(), name.end(),
          [](const char& c) { return ((c == '/') || (c == '\\')); }),
      name.end());
  return name;
}

template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::POISSON>(const double request_rate)
{
  std::exponential_distribution<> dist =
      std::exponential_distribution<>(request_rate);
  return [dist](std::mt19937& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

template <>
std::function<std::chrono::nanoseconds(std::mt19937&)>
ScheduleDistribution<Distribution::CONSTANT>(const double request_rate)
{
  std::chrono::nanoseconds period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / request_rate));
  return [period](std::mt19937& /*gen*/) { return period; };
}

cb::TensorFormat
ParseTensorFormat(const std::string& content_type_str)
{
  std::string content_type_str_lowercase{content_type_str};
  std::transform(
      content_type_str.cbegin(), content_type_str.cend(),
      content_type_str_lowercase.begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (content_type_str_lowercase == "binary") {
    return cb::TensorFormat::BINARY;
  } else if (content_type_str_lowercase == "json") {
    return cb::TensorFormat::JSON;
  } else {
    return cb::TensorFormat::UNKNOWN;
  }
}

std::optional<size_t>
GetDataTypeSize(const std::string& data_type)
{
  if (data_type == "BOOL") {
    return sizeof(bool);
  } else if (data_type == "UINT8") {
    return sizeof(uint8_t);
  } else if (data_type == "UINT16") {
    return sizeof(uint16_t);
  } else if (data_type == "UINT32") {
    return sizeof(uint32_t);
  } else if (data_type == "UINT64") {
    return sizeof(uint64_t);
  } else if (data_type == "INT8") {
    return sizeof(int8_t);
  } else if (data_type == "INT16") {
    return sizeof(int16_t);
  } else if (data_type == "INT32") {
    return sizeof(int32_t);
  } else if (data_type == "INT64") {
    return sizeof(int64_t);
  } else if (data_type == "FP32") {
    return sizeof(float);
  } else if (data_type == "FP64") {
    return sizeof(double);
  } else if (data_type == "BYTES") {
    return sizeof(char);
  } else if (data_type == "JSON") {
    return sizeof(char);
  } else {
    std::cerr << "WARNING: unsupported data type: '" + data_type + "'"
              << std::endl;
    return {};
  }
}

}}  // namespace triton::perfanalyzer
