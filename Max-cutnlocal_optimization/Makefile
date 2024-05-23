CC = g++
CFLAGS=-std=c++11 -Wall

SRCS=task4.cpp
OBJS=$(SRCS:.cpp=.o)
TARGET=task4

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)

run: $(TARGET)
	./$(TARGET)
