CC = g++
CFLAGS=-std=c++11 -Wall

SRCS=birthday.cpp
OBJS=$(SRCS:.cpp=.o)
TARGET=birthday

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)
run: $(TARGET)
	./$(TARGET)
