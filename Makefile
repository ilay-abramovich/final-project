CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

TARGET = symnmf

SRC = symnmf.c
OBJ = $(SRC:.c=.o)

INC= "/usr/include/"

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) -lm

%.o: %.c symnmf.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
