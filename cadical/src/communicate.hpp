#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

struct Connection {
  static constexpr size_t BUFFER_SIZE = 1 << 16;

  int socket_fd;

  explicit Connection(const char *address);

  template <typename T>
  T read() {
    T value;
    ssize_t bytes_read = recv(socket_fd, &value, sizeof(T), 0);
    if (bytes_read != sizeof(T)) {
      std::cerr << "Failed to read value from connection\n";
      exit(1);
    }
    return value;
  }

  uint8_t read_u8() { return read<uint8_t>(); }
  uint16_t read_u16() { return read<uint16_t>(); }
  uint32_t read_u32() { return read<uint32_t>(); }
  uint64_t read_u64() { return read<uint64_t>(); }
  int8_t read_i8() { return read<int8_t>(); }
  int16_t read_i16() { return read<int16_t>(); }
  int32_t read_i32() { return read<int32_t>(); }
  int64_t read_i64() { return read<int64_t>(); }
  float read_f32() { return read<float>(); }
  double read_f64() { return read<double>(); }

  [[nodiscard]] std::string read_raw_string(uint32_t length) const {
    std::string value(length, '\0');
    ssize_t bytes_read = recv(socket_fd, &value[0], length, 0);
    if (bytes_read != length) {
      std::cerr << "Failed to read string from connection\n";
      exit(1);
    }
    return value;
  }

  std::string read_string() {
    uint32_t length = read_u32();
    return read_raw_string(length);
  }

  template <typename T>
  Connection &write(const T &value) {
    ssize_t bytes_written = send(socket_fd, &value, sizeof(T), 0);
    if (bytes_written != sizeof(T)) {
      std::cerr << "Failed to write value to connection\n";
      exit(1);
    }
    return *this;
  }

  Connection &write_u8(uint8_t value) { return write<uint8_t>(value); }
  Connection &write_u16(uint16_t value) { return write<uint16_t>(value); }
  Connection &write_u32(uint32_t value) { return write<uint32_t>(value); }
  Connection &write_u64(uint64_t value) { return write<uint64_t>(value); }
  Connection &write_i8(int8_t value) { return write<int8_t>(value); }
  Connection &write_i16(int16_t value) { return write<int16_t>(value); }
  Connection &write_i32(int32_t value) { return write<int32_t>(value); }
  Connection &write_i64(int64_t value) { return write<int64_t>(value); }
  Connection &write_f32(float value) { return write<float>(value); }
  Connection &write_f64(double value) { return write<double>(value); }

  Connection &write_raw_string(const std::string &value) {
    ssize_t bytes_written = send(socket_fd, value.c_str(), value.size(), 0);
    if (bytes_written != value.size()) {
      std::cerr << "Failed to write string to connection\n";
      exit(1);
    }
    return *this;
  }

  Connection &write_string(const std::string &value) {
    write_u32(value.size());
    return write_raw_string(value);
  }

  void flush() {
  }

  void wait_for_ok();

  void close() {
    flush();
    if(socket_fd >= 0)
      ::close(socket_fd);
  }
};
