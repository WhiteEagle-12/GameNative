#ifndef EVSHIM_MINIMAL_SDL_H
#define EVSHIM_MINIMAL_SDL_H

#include <stdint.h>

#define SDL_INIT_JOYSTICK 0x00000200u
#define SDL_VIRTUAL_JOYSTICK_DESC_VERSION 1

typedef uint8_t Uint8;
typedef uint16_t Uint16;
typedef uint32_t Uint32;
typedef int16_t Sint16;

typedef struct _SDL_Joystick SDL_Joystick;

typedef enum SDL_JoystickType
{
    SDL_JOYSTICK_TYPE_UNKNOWN = 0,
    SDL_JOYSTICK_TYPE_GAMECONTROLLER = 1
} SDL_JoystickType;

typedef struct SDL_version
{
    Uint8 major;
    Uint8 minor;
    Uint8 patch;
} SDL_version;

typedef struct SDL_VirtualJoystickDesc
{
    Uint16 version;
    Uint16 type;
    Uint16 naxes;
    Uint16 nbuttons;
    Uint16 nballs;
    Uint16 nhats;
    Uint16 vendor_id;
    Uint16 product_id;
    Uint16 padding;
    Uint32 button_mask;
    Uint32 axis_mask;
    const char *name;
    void *userdata;
    void (*Update)(void *userdata);
    int (*SetPlayerIndex)(void *userdata, int player_index);
    int (*Rumble)(void *userdata, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble);
    int (*RumbleTriggers)(void *userdata, Uint16 left_rumble, Uint16 right_rumble);
    int (*SetLED)(void *userdata, Uint8 red, Uint8 green, Uint8 blue);
    int (*SendEffect)(void *userdata, const void *data, int size);
} SDL_VirtualJoystickDesc;

#endif
