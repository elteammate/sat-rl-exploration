#include <communicate.hpp>
#include <cassert>

uint8_t Connection::read_u8() {
  uint8_t value;
  fread(&value, sizeof(uint8_t), 1, pipe_in);
  return value;
}

uint16_t Connection::read_u16() {
  uint16_t value;
  fread(&value, sizeof(uint16_t), 1, pipe_in);
  return value;
}

uint32_t Connection::read_u32() {
  uint32_t value;
  fread(&value, sizeof(uint32_t), 1, pipe_in);
  return value;
}

uint64_t Connection::read_u64() {
  uint64_t value;
  fread(&value, sizeof(uint64_t), 1, pipe_in);
  return value;
}

int8_t Connection::read_i8() {
  int8_t value;
  fread(&value, sizeof(int8_t), 1, pipe_in);
  return value;
}

int16_t Connection::read_i16() {
  int16_t value;
  fread(&value, sizeof(int16_t), 1, pipe_in);
  return value;
}

int32_t Connection::read_i32() {
  int32_t value;
  fread(&value, sizeof(int32_t), 1, pipe_in);
  return value;
}

int64_t Connection::read_i64() {
  int64_t value;
  fread(&value, sizeof(int64_t), 1, pipe_in);
  return value;
}

std::string Connection::read_raw_string(uint32_t length) {
  std::string value(length, '\0');
  fread(&value[0], sizeof(char), length, pipe_in);
  return value;
}

void Connection::write_u8(uint8_t value) {
  fwrite(&value, sizeof(uint8_t), 1, pipe_out);
}

void Connection::write_u16(uint16_t value) {
  fwrite(&value, sizeof(uint16_t), 1, pipe_out);
}

void Connection::write_u32(uint32_t value) {
  fwrite(&value, sizeof(uint32_t), 1, pipe_out);
}

void Connection::write_u64(uint64_t value) {
  fwrite(&value, sizeof(uint64_t), 1, pipe_out);
}

void Connection::write_i8(int8_t value) {
  fwrite(&value, sizeof(int8_t), 1, pipe_out);
}

void Connection::write_i16(int16_t value) {
  fwrite(&value, sizeof(int16_t), 1, pipe_out);
}

void Connection::write_i32(int32_t value) {
  fwrite(&value, sizeof(int32_t), 1, pipe_out);
}

void Connection::write_i64(int64_t value) {
  fwrite(&value, sizeof(int64_t), 1, pipe_out);
}

void Connection::write_raw_string(const std::string &value) {
  fwrite(&value[0], sizeof(char), value.size(), pipe_out);
}

void Connection::write_string(const std::string &value) {
  write_u32(value.size());
  write_raw_string(value);
}

void Connection::flush() {
  fflush(pipe_out);
}

void Connection::close() {
  fclose(pipe_in);
  fclose(pipe_out);
}

std::string Connection::read_string() {
  uint32_t length = read_u32();
  assert(length < 256 && "I'm unsure if you ever need to read strings that large");
  return read_raw_string(length);
}

float Connection::read_f32() {
  float value;
  fread(&value, sizeof(value), 1, pipe_in);
  return value;
}

double Connection::read_f64() {
  double value;
  fread(&value, sizeof(value), 1, pipe_in);
  return value;
}

void Connection::write_f32(float value) {
  fwrite(&value, sizeof(value), 1, pipe_out);
}

void Connection::write_f64(double value) {
  fwrite(&value, sizeof(value), 1, pipe_out);
}

void Connection::wait_for_ok() {
  assert(read_string() == "ok" && "Expected to read ok from connection");
}
