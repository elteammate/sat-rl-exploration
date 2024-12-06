#include <cassert>
#include <communicate.hpp>
#include "internal.hpp"

namespace CaDiCaL {
void Connection::wait_for_ok() {
  auto message = read_string();
  if (message != "ok") {
    std::cerr << "Expected to read ok from connection, but got " << message << std::endl;
    exit(1);
  }
}

Connection::Connection(const char *address) {
  socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if(socket_fd == -1) {
    std::cerr << "Failed to create socket\n" << errno << std::endl;
    exit(1);
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, address, sizeof(addr.sun_path) - 1);

  // Connect to the socket
  if(connect(socket_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
    ::close(socket_fd);
    std::cerr << "Failed to connect to socket\n" << errno << std::endl;
    exit(1);
  }
}

Connection &Connection::write_clause(const Clause &c) {
  write_u64(c.id);
  uint64_t flags = 0;
  if (c.redundant) flags |= 1;
  if (c.keep) flags |= 2;
  if (c.used == 2) flags |= 4;
  write_u64(flags);
  write_i32(c.glue);
  write_u32(c.size);
  for(auto l : c) {
    write_i32(l);
  }

  return *this;
}

}