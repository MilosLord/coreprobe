CXX      ?= g++
CXXFLAGS ?= -O0 -std=c++20 -Wall -Wextra -Wpedantic
LDFLAGS  ?= -pthread

PREFIX   ?= /usr/local

all: coreprobe

coreprobe: coreprobe.cpp
	$(CXX) $(CXXFLAGS) -o $@ coreprobe.cpp $(LDFLAGS)

windows-static: coreprobe.cpp
	$(CXX) $(CXXFLAGS) -static -static-libgcc -static-libstdc++ -o coreprobe.exe coreprobe.cpp

install: coreprobe
	install -D -m 0755 coreprobe $(DESTDIR)$(PREFIX)/bin/coreprobe

clean:
	rm -f coreprobe coreprobe.exe coreprobe_results.json coreprobe_results.json.tmp

.PHONY: all windows-static install clean
