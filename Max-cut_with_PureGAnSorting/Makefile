CXX = g++
CXXFLAGS = -std=gnu++17 -O3 -Wall

TARGET = GA3

all: $(TARGET)

$(TARGET): GA3.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) GA3.cpp

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	./$(TARGET)
