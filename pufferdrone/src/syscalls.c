// pufferdrone/src/syscalls.c
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>

int _close(int fd)                { (void)fd; return 0; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd)               { (void)fd; return 1; }
off_t _lseek(int fd, off_t o, int w){ (void)fd; (void)o; (void)w; return 0; }
int _read(int fd, void *buf, size_t cnt) { (void)fd; (void)buf; (void)cnt; return 0; }

// TODO: wire this to your console/UART if you want printf() output
int _write(int fd, const void *buf, size_t cnt) {
  (void)fd; (void)buf;
  return (int)cnt; // pretend we wrote everything
}

// If you want to allow newlib heap, implement a real _sbrk using your linker symbols.
// For now, disallow heap to avoid accidental malloc().
void * _sbrk(ptrdiff_t incr) {
  errno = ENOMEM;
  return (void*)-1;
}
