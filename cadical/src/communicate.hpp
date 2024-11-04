#pragma once

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>


struct Connection {
  FILE *pipe_in;
  FILE *pipe_out;

  Connection(FILE *pipe_in, FILE *pipe_out) : pipe_in(pipe_in), pipe_out(pipe_out) {}

  uint8_t read_u8();

  uint16_t read_u16();

  uint32_t read_u32();

  uint64_t read_u64();

  int8_t read_i8();

  int16_t read_i16();

  int32_t read_i32();

  int64_t read_i64();

  float read_f32();

  double read_f64();

  std::string read_raw_string(uint32_t length);

  std::string read_string();

  void write_u8(uint8_t value);

  void write_u16(uint16_t value);

  void write_u32(uint32_t value);

  void write_u64(uint64_t value);

  void write_i8(int8_t value);

  void write_i16(int16_t value);

  void write_i32(int32_t value);

  void write_i64(int64_t value);

  void write_f32(float value);

  void write_f64(double value);

  void write_raw_string(const std::string &value);

  void write_string(const std::string &value);

  void flush();

  void close();

  void wait_for_ok();
};
