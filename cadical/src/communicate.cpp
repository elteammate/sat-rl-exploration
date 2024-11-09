#include <cassert>
#include <communicate.hpp>


void Connection::wait_for_ok() {
  assert(read_string() == "ok" && "Expected to read ok from connection");
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
