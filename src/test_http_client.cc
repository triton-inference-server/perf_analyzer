// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
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

#include <curl/curl.h>

#include <iostream>
#include <sstream>

#include "client_backend/openai/openai_client.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

class TestHTTPClient : public HttpClient {
 public:
  using HttpClient::PrintCurlCommand;

  TestHTTPClient(
      const std::string& url, bool verbose,
      const HttpSslOptions& ssl_options = HttpSslOptions())
      : HttpClient(url, verbose, ssl_options)
  {
  }
};

std::string
capture_cout(std::function<void()> func)
{
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
  func();
  std::cout.rdbuf(old);
  return buffer.str();
}

TEST_CASE("Test PrintCurlCommand")
{
  const std::string url = "https://api.example.com/v1/chat/completions";
  CURL* curl_handle = curl_easy_init();
  curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());

  const char* test_data = "{\"prompt\":\"Hello, world!\"}";
  auto request = std::make_unique<HttpRequest>(
      [](HttpRequest*) { /* completion callback */ }, true);
  request->AddInput(
      reinterpret_cast<uint8_t*>(const_cast<char*>(test_data)),
      strlen(test_data));

  struct curl_slist* header_list = nullptr;
  header_list =
      curl_slist_append(header_list, "Authorization: Bearer test_token");
  header_list =
      curl_slist_append(header_list, "Content-Type: application/json");
  request->header_list_ = header_list;

  SUBCASE("PrintCurlCommand when verbose is false")
  {
    TestHTTPClient test_client(url, false);
    std::string output = capture_cout([&]() {
      test_client.PrintCurlCommand(curl_handle, std::move(request));
    });
    CHECK(output.empty());
  }

  SUBCASE("PrintCurlCommand when verbose is true")
  {
    TestHTTPClient test_client(url, true);
    std::string output = capture_cout([&]() {
      test_client.PrintCurlCommand(curl_handle, std::move(request));
    });

    CHECK(!output.empty());
    CHECK(output.find("curl") != std::string::npos);
    CHECK(output.find(url) != std::string::npos);
    CHECK(
        output.find("-H \"Authorization: Bearer test_token\"") !=
        std::string::npos);
    CHECK(
        output.find("-H \"Content-Type: application/json\"") !=
        std::string::npos);
    CHECK(
        output.find("-d '{\"prompt\":\"Hello, world!\"}'") !=
        std::string::npos);
  }

  curl_easy_cleanup(curl_handle);
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
