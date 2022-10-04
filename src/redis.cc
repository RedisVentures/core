
#pragma once

#include <list>
#include <string>
#include <iostream>
#include <string_view>
#include <unordered_map>

#include <sw/redis++/redis++.h>



std::unique_ptr<sw::redis::RedisCluster> init_cluster_client(
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
  std::unique_ptr<sw::redis::RedisCluster> redis = std::make_unique<sw::redis::RedisCluster>(options, pool_options);
  return redis;
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

void cache_set(std::unique_ptr<sw::redis::Redis> &redis, RedisCacheEntry<std::string_view> &cache_entry) {

    //use hash_tag here
    std::string prefix = "{" + std::string(cache_entry.key.data()) + "}";

    // set number of entries in the top level
    redis->set(prefix, std::to_string(cache_entry.num_entries));

    int i = 1;
    for (auto &output: cache_entry.outputs) {
      std::string key = prefix + "." + std::to_string(i);
      redis->hmset(key, output.fields.begin(), output.fields.end());
      i += 1;
    }
}

RedisCacheEntry<std::string> cache_get(
    std::unique_ptr<sw::redis::Redis> &redis,
    std::string_view &hash_key) {

    RedisCacheEntry<std::string> entry;

    //use hash_tag here
    std::string prefix = "{" + std::string(hash_key.data()) + "}";

    std::string num = redis->get(prefix).value();
    int num_entries = std::atoi(num.data());

    std::vector<HashEntry<std::string>> entries;
    /// TODO pipeline these
    for (int i=1; i <=num_entries; i++) {
      HashEntry<std::string> h;
      std::string key = prefix + "." + std::to_string(i);
      redis->hgetall(key, std::inserter(h.fields, h.fields.begin()));
      entries.emplace_back(h);
    }

    entry.key = prefix;
    entry.num_entries = num_entries;
    entry.outputs = entries;

    return entry;
}
