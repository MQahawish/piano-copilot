#!/usr/bin/env python

import argparse
import codecs
import os
import re
import shutil
import warnings
import time
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import pygame.gfxdraw
from pygame import Surface, Rect
import math
import keyboardlayout as kl
import keyboardlayout.pygame as klp
import librosa
import numpy
import pygame
import soundfile
import mido  # Added MIDI library
import sys
ANCHOR_INDICATOR = " anchor"
ANCHOR_NOTE_REGEX = re.compile(r"\s[abcdefg]$")
DESCRIPTION = 'Use your computer keyboard as a "piano"'
DESCRIPTOR_32BIT = "FLOAT"
BITS_32BIT = 32
AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED = 0
SOUND_FADE_MILLISECONDS = 50
CYAN = (0, 255, 255, 255)
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)

AUDIO_ASSET_PREFIX = "audio_files/"
KEYBOARD_ASSET_PREFIX = "keyboards/"
RECORDINGS_FOLDER = "recordings/"
CURRENT_WORKING_DIR = Path(__file__).parent.absolute()
ALLOWED_EVENTS = {pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT, pygame.MOUSEBUTTONDOWN}

# MIDI Constants
MIDI_C4 = 60  # MIDI note number for middle C
MIDI_VELOCITY = 64  # Default velocity for MIDI notes
MIDI_TEMPO = 500000  # Microseconds per beat (500000 = 120 BPM)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    default_wav_file = "audio_files/piano_c4.wav"
    parser.add_argument(
        "--wav",
        "-w",
        metavar="FILE",
        type=str,
        default=default_wav_file,
        help="WAV file (default: {})".format(default_wav_file),
    )
    default_keyboard_file = "keyboards/azerty_typewriter.txt"
    parser.add_argument(
        "--keyboard",
        "-k",
        metavar="FILE",
        type=str,
        default=default_keyboard_file,
        help="keyboard file (default: {})".format(default_keyboard_file),
    )
    parser.add_argument(
        "--clear-cache",
        "-c",
        default=False,
        action="store_true",
        help="deletes stored transposed audio files and recalculates them",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode")
    
    return parser


def get_or_create_key_sounds(
    wav_path: str,
    sample_rate_hz: int,
    channels: int,
    tones: List[int],
    clear_cache: bool,
    keys: List[str],
) -> Generator[pygame.mixer.Sound, None, None]:
    sounds = []
    y, sr = librosa.load(wav_path, sr=sample_rate_hz, mono=channels == 1)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    folder_containing_wav = Path(wav_path).parent.absolute()
    cache_folder_path = Path(folder_containing_wav, file_name)
    if clear_cache and cache_folder_path.exists():
        shutil.rmtree(cache_folder_path)
    if not cache_folder_path.exists():
        print("Generating samples for each key")
        os.mkdir(cache_folder_path)
    for i, tone in enumerate(tones):
        cached_path = Path(cache_folder_path, "{}.wav".format(tone))
        if Path(cached_path).exists():
            print("Loading note {} out of {} for {}".format(i + 1, len(tones), keys[i]))
            sound, sr = librosa.load(cached_path, sr=sample_rate_hz, mono=channels == 1)
            if channels > 1:
                # the shape must be [length, 2]
                sound = numpy.transpose(sound)
        else:
            print(
                "Transposing note {} out of {} for {}".format(
                    i + 1, len(tones), keys[i]
                )
            )
            if channels == 1:
                # Fixed pitch_shift call to match current librosa API
                sound = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=tone)
            else:
                new_channels = [
                    # Fixed pitch_shift call to match current librosa API
                    librosa.effects.pitch_shift(y=y[i], sr=sr, n_steps=tone)
                    for i in range(channels)
                ]
                sound = numpy.ascontiguousarray(numpy.vstack(new_channels).T)
            soundfile.write(cached_path, sound, sample_rate_hz, DESCRIPTOR_32BIT)
        sounds.append(sound)
    sounds = map(pygame.sndarray.make_sound, sounds)
    return sounds


BLACK_INDICES_C_SCALE = [1, 3, 6, 8, 10]
LETTER_KEYS_TO_INDEX = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def __get_black_key_indices(key_name: str) -> set:
    letter_key_index = LETTER_KEYS_TO_INDEX[key_name]
    black_key_indices = set()
    for ind in BLACK_INDICES_C_SCALE:
        new_index = ind - letter_key_index
        if new_index < 0:
            new_index += 12
        black_key_indices.add(new_index)
    return black_key_indices


def get_keyboard_info(keyboard_file: str):
    with codecs.open(keyboard_file, encoding="utf-8") as k_file:
        lines = k_file.readlines()
    keys = []
    anchor_index = -1
    black_key_indices = set()
    anchor_note = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        match = ANCHOR_NOTE_REGEX.search(line)
        if match:
            anchor_index = i
            anchor_note = line[-1]  # Get the anchor note letter
            black_key_indices = __get_black_key_indices(anchor_note)
            key = kl.Key(line[: match.start(0)])
        elif line.endswith(ANCHOR_INDICATOR):
            anchor_index = i
            key = kl.Key(line[: -len(ANCHOR_INDICATOR)])
        else:
            key = kl.Key(line)
        keys.append(key)
    if anchor_index == -1:
        raise ValueError(
            "Invalid keyboard file, one key must have an anchor note or the "
            "word anchor written next to it.\n"
            "For example 'm c OR m anchor'.\n"
            "That tells the program that the wav file will be used for key m, "
            "and all other keys will be pitch shifted higher or lower from "
            "that anchor. If an anchor note is used then keys are colored black "
            "and white like a piano. If the word anchor is used, then the "
            "highest key is white, and keys get darker as they descend in pitch."
        )
    tones = [i - anchor_index for i in range(len(keys))]
    color_to_key = defaultdict(list)
    if black_key_indices:
        key_color = (120, 120, 120, 255)
        key_txt_color = (50, 50, 50, 255)
    else:
        key_color = (65, 65, 65, 255)
        key_txt_color = (0, 0, 0, 255)
    for index, key in enumerate(keys):
        if index == anchor_index:
            color_to_key[CYAN].append(key)
            continue
        if black_key_indices:
            used_index = (index - anchor_index) % 12
            if used_index in black_key_indices:
                color_to_key[BLACK].append(key)
                continue
            color_to_key[WHITE].append(key)
            continue
        # anchor mode, keys go up in half steps and we do not color black keys
        # instead we color from grey low to white high
        rgb_val = 255 - (len(keys) - 1 - index) * 3
        color = (rgb_val, rgb_val, rgb_val, 255)
        color_to_key[color].append(key)

    return keys, tones, color_to_key, key_color, key_txt_color, anchor_note


class Button:
    def __init__(self, x, y, width, height, color, text, icon=None, text_color=WHITE, 
                 font_size=16, border_radius=8):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = self._get_hover_color(color)
        self.active_color = self._get_active_color(color)
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.border_radius = border_radius
        self.icon = icon
        self.is_active = False
        self.is_hovering = False
        self.shadow_offset = 3
        
    def _get_hover_color(self, color):
        """Return a slightly lighter version of the color for hover state"""
        r, g, b = color[:3]
        factor = 1.1
        return (min(int(r * factor), 255), 
                min(int(g * factor), 255), 
                min(int(b * factor), 255), 
                color[3] if len(color) > 3 else 255)
    
    def _get_active_color(self, color):
        """Return a slightly darker version of the color for active state"""
        r, g, b = color[:3]
        factor = 0.8
        return (int(r * factor), 
                int(g * factor), 
                int(b * factor), 
                color[3] if len(color) > 3 else 255)
        
    def draw(self, screen):
        # Determine current color based on state
        current_color = self.active_color if self.is_active else (
                       self.hover_color if self.is_hovering else self.color)
        
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += self.shadow_offset
        self._draw_rounded_rect(screen, shadow_rect, (30, 30, 30, 180), self.border_radius)
        
        # Draw button
        self._draw_rounded_rect(screen, self.rect, current_color, self.border_radius)
        
        # Draw button border
        pygame.draw.rect(screen, (200, 200, 200, 100), self.rect, 1, border_radius=self.border_radius)
        
        # Calculate text position
        text_surface = self.font.render(self.text, True, self.text_color)
        
        if self.icon:
            # If there's an icon, position text to its right
            icon_size = 24
            icon_padding = 8
            total_width = icon_size + icon_padding + text_surface.get_width()
            
            icon_x = self.rect.centerx - total_width // 2
            icon_y = self.rect.centery - icon_size // 2
            
            # Draw icon
            self._draw_icon(screen, icon_x, icon_y, icon_size, self.icon)
            
            # Draw text
            text_x = icon_x + icon_size + icon_padding
            text_y = self.rect.centery - text_surface.get_height() // 2
            screen.blit(text_surface, (text_x, text_y))
        else:
            # Center text if no icon
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
    
    def _draw_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners"""
        if radius <= 0:
            pygame.draw.rect(surface, color, rect)
            return
            
        # Get the rectangle dimensions
        x, y, width, height = rect
        
        # Draw the rectangle with rounded corners
        pygame.gfxdraw.box(surface, (x + radius, y, width - 2 * radius, height), color)
        pygame.gfxdraw.box(surface, (x, y + radius, width, height - 2 * radius), color)
        
        # Draw the four rounded corners
        pygame.gfxdraw.filled_circle(surface, x + radius, y + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + width - radius - 1, y + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + radius, y + height - radius - 1, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + width - radius - 1, y + height - radius - 1, radius, color)
    
    def _draw_icon(self, screen, x, y, size, icon_type):
        """Draw different icons based on type"""
        if icon_type == "record":
            # Drawing a record circle
            pygame.gfxdraw.filled_circle(screen, x + size//2, y + size//2, size//2 - 2, (255, 0, 0))
            pygame.gfxdraw.aacircle(screen, x + size//2, y + size//2, size//2 - 2, (255, 0, 0))
        
        elif icon_type == "stop":
            # Drawing a stop square
            stop_rect = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
            pygame.draw.rect(screen, WHITE, stop_rect)
        
        elif icon_type == "save_wav":
            # Drawing a waveform icon
            wave_points = []
            for i in range(size):
                wave_points.append((x + i, y + size//2 + int(math.sin(i/2) * size//4)))
            
            pygame.draw.lines(screen, WHITE, False, wave_points, 2)
        
        elif icon_type == "save_midi":
            # Drawing a simple piano keys icon
            key_width = size // 4
            white_key_height = size - 6
            black_key_height = white_key_height * 0.6
            
            # Draw white keys
            for i in range(3):
                pygame.draw.rect(screen, WHITE, 
                                (x + i * key_width + 2, y + 3, 
                                 key_width - 1, white_key_height))
            
            # Draw black keys
            pygame.draw.rect(screen, (40, 40, 40), 
                            (x + key_width * 0.7, y + 3, 
                             key_width * 0.6, black_key_height))
            pygame.draw.rect(screen, (40, 40, 40), 
                            (x + key_width * 1.7, y + 3, 
                             key_width * 0.6, black_key_height))
        
    def update(self, mouse_pos):
        """Update button state based on mouse position"""
        self.is_hovering = self.rect.collidepoint(mouse_pos)
        
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class UIPanel:
    def __init__(self, x, y, width, height, bg_color=(30, 30, 30)):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.elements = []
        self.title = None
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True)
        
    def set_title(self, title, color=WHITE):
        self.title = title
        self.title_color = color
        
    def add_element(self, element):
        self.elements.append(element)
        
    def draw(self, screen):
        # Draw panel background with rounded corners
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=10)
        
        # Draw panel border
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 1, border_radius=10)
        
        # Draw title if set
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(midtop=(self.rect.centerx, self.rect.y + 5))
            screen.blit(title_surface, title_rect)
        
        # Draw all contained elements
        for element in self.elements:
            element.draw(screen)

# Status message class for more elegant notifications
class StatusMessage:
    def __init__(self):
        self.message = ""
        self.color = WHITE
        self.font = pygame.font.SysFont("Arial", 18, bold=True)  # Larger font size and bold
        self.start_time = 0
        self.duration = 4  # seconds to display the message
        self.fade_duration = 0.5  # seconds to fade out
        self.background_color = (30, 30, 35, 180)  # Semi-transparent background
        
    def set_message(self, message, color=WHITE):
        self.message = message
        self.color = color
        self.start_time = time.time()
        
    def draw(self, screen, position):
        if not self.message:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # If the message has expired, clear it
        if elapsed > self.duration:
            self.message = ""
            return
            
        # Calculate opacity for fade out
        alpha = 255
        if elapsed > (self.duration - self.fade_duration):
            fade_percent = (elapsed - (self.duration - self.fade_duration)) / self.fade_duration
            alpha = int(255 * (1 - fade_percent))
        
        # Create a surface with per-pixel alpha
        text_surface = self.font.render(self.message, True, self.color)
        text_rect = text_surface.get_rect()
        
        # Create a background surface with padding
        padding = 10
        bg_rect = pygame.Rect(
            position[0] - padding // 2,
            position[1] - padding // 2,
            text_rect.width + padding,
            text_rect.height + padding
        )
        
        # Draw background with rounded corners
        background = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            background, 
            self.background_color, 
            (0, 0, bg_rect.width, bg_rect.height), 
            border_radius=5
        )
        background.set_alpha(alpha)
        screen.blit(background, bg_rect.topleft)
        
        # Apply the calculated alpha to text
        text_surface.set_alpha(alpha)
        
        # Draw to screen
        screen.blit(text_surface, position)

class MIDIPlayer:
    def __init__(self, status_callback=None):
        self.current_midi = None
        self.playing = False
        self.status_callback = status_callback
        self.process = None
        
    def load_midi(self, midi_path):
        """Load a MIDI file for playback"""
        if not os.path.exists(midi_path):
            if self.status_callback:
                self.status_callback("MIDI file not found", (255, 100, 100))
            return False
            
        self.current_midi = midi_path
        return True
        
    def play(self):
        """Start playback of loaded MIDI file"""
        if not self.current_midi or not os.path.exists(self.current_midi):
            if self.status_callback:
                self.status_callback("No MIDI file loaded", (255, 100, 100))
            return False
            
        if self.playing:
            return True
            
        # Play MIDI using an external player (platform dependent)
        # This is a simple implementation - in production you might want to use
        # a proper MIDI library that integrates better with pygame
        try:
            import subprocess
            
            if sys.platform == 'win32':
                self.process = subprocess.Popen(['start', 'wmplayer', self.current_midi], 
                                               shell=True)
            elif sys.platform == 'darwin':
                self.process = subprocess.Popen(['afplay', self.current_midi])
            else:
                self.process = subprocess.Popen(['timidity', self.current_midi])
                
            self.playing = True
            
            if self.status_callback:
                self.status_callback(f"Playing {os.path.basename(self.current_midi)}", (100, 255, 100))
            return True
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Playback error: {str(e)}", (255, 100, 100))
            return False
            
    def pause(self):
        """Pause current playback"""
        if not self.playing or not self.process:
            return
            
        try:
            if self.process:
                self.process.terminate()
                self.process = None
                
            self.playing = False
            
            if self.status_callback:
                self.status_callback("Playback paused", (255, 200, 100))
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error pausing: {str(e)}", (255, 100, 100))

    # Create slider for temperature control
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, integer_only=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_radius = height + 4  # Slightly larger handle
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.integer_only = integer_only  # New flag to control integer-only mode
        self.font = pygame.font.SysFont("Arial", 16)
        self.value_font = pygame.font.SysFont("Arial", 14)
        self.dragging = False
        self.track_color = (60, 60, 70)
        self.track_active_color = (80, 100, 140)
        self.handle_color = (120, 140, 220)
        self.handle_hover_color = (140, 160, 240)
        self.text_color = WHITE
        self.is_hovered = False
        
        # If integer-only mode is enabled, ensure the initial value is an integer
        if self.integer_only:
            self.value = int(self.value)
        
        # Calculate handle position
        self.handle_x = self._value_to_pos(initial_val)
        
    def _value_to_pos(self, value):
        """Convert slider value to handle x position"""
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(normalized * self.rect.width)
        
    def _pos_to_value(self, pos_x):
        """Convert handle x position to slider value"""
        pos = max(self.rect.x, min(pos_x, self.rect.x + self.rect.width))
        normalized = (pos - self.rect.x) / self.rect.width
        value = self.min_val + normalized * (self.max_val - self.min_val)
        
        # If integer-only mode is enabled, round to the nearest integer
        if self.integer_only:
            value = int(round(value))
            
        return value
        
    def draw(self, screen):
        # Draw background track (inactive part)
        pygame.draw.rect(screen, self.track_color, self.rect, border_radius=self.rect.height//2)
        
        # Draw active part of the track
        active_width = self.handle_x - self.rect.x
        if active_width > 0:
            active_rect = pygame.Rect(self.rect.x, self.rect.y, active_width, self.rect.height)
            pygame.draw.rect(screen, self.track_active_color, active_rect, border_radius=self.rect.height//2)
        
        # Draw label above slider with value
        label_text = self.font.render(f"{self.label}", True, self.text_color)
        
        # Format value based on integer-only mode
        if self.integer_only:
            value_text = self.value_font.render(f"{self.value}", True, (180, 200, 255))
        else:
            value_text = self.value_font.render(f"{self.value:.2f}", True, (180, 200, 255))
        
        # Position label centered above slider
        label_x = self.rect.x + (self.rect.width - label_text.get_width()) // 2
        screen.blit(label_text, (label_x, self.rect.y - 25))
        
        # Position value text below the label
        value_x = self.rect.x + (self.rect.width - value_text.get_width()) // 2
        screen.blit(value_text, (value_x, self.rect.y - 5))
        
        # Draw min/max values as smaller text
        # Format based on integer-only mode
        if self.integer_only:
            min_text = self.value_font.render(f"{int(self.min_val)}", True, (150, 150, 150))
            max_text = self.value_font.render(f"{int(self.max_val)}", True, (150, 150, 150))
        else:
            min_text = self.value_font.render(f"{self.min_val}", True, (150, 150, 150))
            max_text = self.value_font.render(f"{self.max_val}", True, (150, 150, 150))
            
        screen.blit(min_text, (self.rect.x - 5, self.rect.y + self.rect.height + 5))
        screen.blit(max_text, (self.rect.x + self.rect.width - 10, self.rect.y + self.rect.height + 5))
        
        # Draw handle with hover effect
        handle_color = self.handle_hover_color if self.is_hovered or self.dragging else self.handle_color
        pygame.draw.circle(screen, handle_color, (self.handle_x, self.rect.centery), self.handle_radius)
        
        # Draw handle border for better visibility
        pygame.draw.circle(screen, (255, 255, 255, 100), (self.handle_x, self.rect.centery), 
                         self.handle_radius, 1)
        
    def update(self, mouse_pos):
        # Check if mouse is hovering over handle
        mouse_x, mouse_y = mouse_pos
        handle_rect = pygame.Rect(self.handle_x - self.handle_radius, 
                                 self.rect.centery - self.handle_radius,
                                 self.handle_radius * 2, 
                                 self.handle_radius * 2)
        self.is_hovered = handle_rect.collidepoint(mouse_x, mouse_y)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if clicked on handle or track
            mouse_x, mouse_y = event.pos
            handle_rect = pygame.Rect(self.handle_x - self.handle_radius, 
                                     self.rect.centery - self.handle_radius,
                                     self.handle_radius * 2, 
                                     self.handle_radius * 2)
                                     
            # Allow clicking anywhere on the track to move handle
            track_rect = pygame.Rect(
                self.rect.x - self.handle_radius, 
                self.rect.y - self.handle_radius,
                self.rect.width + self.handle_radius * 2,
                self.rect.height + self.handle_radius * 2
            )
            
            if handle_rect.collidepoint(mouse_x, mouse_y) or track_rect.collidepoint(mouse_x, mouse_y):
                self.dragging = True
                # Immediately update handle position when clicked
                self.handle_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
                self.value = self._pos_to_value(self.handle_x)
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                # Update handle position to match integer value for integer-only sliders
                if self.integer_only:
                    self.handle_x = self._value_to_pos(self.value)
                return True
            
        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            self.update(event.pos)
            
            # Update position if dragging
            if self.dragging:
                self.handle_x = max(self.rect.x, min(event.pos[0], self.rect.x + self.rect.width))
                self.value = self._pos_to_value(self.handle_x)
                return True
            
        return False

def configure_pygame_audio_and_set_ui(
    framerate_hz: int,
    channels: int,
    keyboard_arg: str,
    color_to_key: Dict[str, List[kl.Key]],
    key_color: Tuple[int, int, int, int],
    key_txt_color: Tuple[int, int, int, int],
) -> Tuple[pygame.Surface, klp.KeyboardLayout, List[Button], UIPanel, UIPanel, StatusMessage, Slider, Slider]:
    # ui
    pygame.display.init()
    pygame.display.set_caption("Pianoputer AI")

    # block events that we don't want, this must be after display.init
    pygame.event.set_blocked(None)
    pygame.event.set_allowed(list(ALLOWED_EVENTS))

    # fonts
    pygame.font.init()

    # audio
    pygame.mixer.init(
        framerate_hz,
        BITS_32BIT,
        channels,
        allowedchanges=AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED,
    )

    if "qwerty" in keyboard_arg:
        layout_name = kl.LayoutName.QWERTY
    elif "azerty" in keyboard_arg:
        layout_name = kl.LayoutName.AZERTY_LAPTOP
    else:
        ValueError("keyboard must have qwerty or azerty in its name")
    margin = 4
    key_size = 60
    overrides = {}
    for color_value, keys in color_to_key.items():
        override_color = color = pygame.Color(color_value)
        inverted_color = list(~override_color)
        other_val = 255
        if (
            abs(color_value[0] - inverted_color[0]) > abs(color_value[0] - other_val)
        ) or color_value == CYAN:
            override_txt_color = pygame.Color(inverted_color)
        else:
            # biases grey override keys to use white as txt_color
            override_txt_color = pygame.Color([other_val] * 3 + [255])
        override_key_info = kl.KeyInfo(
            margin=margin,
            color=override_color,
            txt_color=override_txt_color,
            txt_font=pygame.font.SysFont("Arial", key_size // 4),
            txt_padding=(key_size // 10, key_size // 10),
        )
        for key in keys:
            overrides[key.value] = override_key_info

    key_txt_color = pygame.Color(key_txt_color)
    keyboard_info = kl.KeyboardInfo(position=(0, 0), padding=2, color=key_txt_color)
    key_info = kl.KeyInfo(
        margin=margin,
        color=pygame.Color(key_color),
        txt_color=pygame.Color(key_txt_color),
        txt_font=pygame.font.SysFont("Arial", key_size // 4),
        txt_padding=(key_size // 6, key_size // 10),
    )
    letter_key_size = (key_size, key_size)  # width, height
    keyboard = klp.KeyboardLayout(
        layout_name, keyboard_info, letter_key_size, key_info, overrides
    )
    
    # Panel dimensions
    control_panel_height = 120
    ai_panel_height = 300  # Make taller to fit sliders and controls properly
    
    screen_width = max(keyboard.rect.width, 800)  # Ensure minimum width
    screen_height = keyboard.rect.height + control_panel_height + ai_panel_height + 30  # Extra spacing
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    
    # Dark background
    screen.fill((20, 20, 25))
    
    # Create UI panel for recording controls
    panel = UIPanel(10, keyboard.rect.height + 10, screen_width - 20, control_panel_height - 20, 
                    bg_color=(40, 40, 45))
    panel.set_title("Recording Controls")
    
    # Define button dimensions and spacing
    button_width = 120
    button_height = 40
    button_spacing = (screen_width - 60 - (button_width * 3)) // 2  # Calculate equal spacing
    start_x = 30  # Start from left with padding
    button_y_offset = keyboard.rect.height + 45  # Y position for recording controls
    
    # Create recording control buttons
    record_button = Button(
        start_x, 
        button_y_offset, 
        button_width, 
        button_height, 
        (220, 60, 60, 255),  # Red
        "RECORD",
        icon="record"
    )
    
    stop_button = Button(
        start_x + button_width + button_spacing, 
        button_y_offset, 
        button_width, 
        button_height, 
        (70, 70, 80, 255),  # Dark gray
        "STOP",
        icon="stop",
        text_color=WHITE
    )
    
    save_midi_button = Button(
        start_x + (button_width + button_spacing) * 2, 
        button_y_offset, 
        button_width, 
        button_height, 
        (70, 105, 200, 255),  # Blue
        "SAVE MIDI",
        icon="save_midi"
    )
    
    # Add recording buttons to panel
    panel.add_element(record_button)
    panel.add_element(stop_button)
    panel.add_element(save_midi_button)
    
    # Calculate AI panel position (below recording panel)
    ai_panel_y = keyboard.rect.height + control_panel_height + 10
    
    # Create AI control panel
    ai_panel = UIPanel(
        10, 
        ai_panel_y, 
        screen_width - 20, 
        ai_panel_height - 20, 
        bg_color=(35, 40, 50)
    )
    ai_panel.set_title("AI Composition")
    
    # First row of AI controls (Generate, Accept, Retry)
    ai_button_y = ai_panel_y + 45  # First row of AI buttons
    
    generate_button = Button(
        start_x, 
        ai_button_y,
        button_width, 
        button_height, 
        (100, 130, 220, 255),  # Blue
        "GENERATE",
        text_color=WHITE
    )
    
    accept_button = Button(
        start_x + button_width + button_spacing, 
        ai_button_y,
        button_width, 
        button_height, 
        (100, 220, 100, 255),  # Green
        "ACCEPT",
        text_color=WHITE
    )
    
    retry_button = Button(
        start_x + (button_width + button_spacing) * 2, 
        ai_button_y,
        button_width, 
        button_height, 
        (220, 160, 40, 255),  # Orange
        "RETRY",
        text_color=WHITE
    )
    
    # Second row - Playback controls (Play/Pause side by side)
    playback_y = ai_button_y + button_height + 20  # Second row with 20px spacing
    
    play_button = Button(
        start_x, 
        playback_y,
        button_width // 2, 
        button_height, 
        (100, 180, 100, 255),  # Green
        "PLAY",
        text_color=WHITE
    )
    
    pause_button = Button(
        start_x + button_width // 2, 
        playback_y,
        button_width // 2, 
        button_height, 
        (70, 70, 80, 255),  # Dark gray
        "PAUSE",
        text_color=WHITE
    )
    
    # Revised slider positions - make sure they're centered and properly spaced
    slider_width = screen_width - 140  # Leave margins on sides
    slider_x = (screen_width - slider_width) // 2  # Center horizontally
    
    # Position sliders with enough spacing from top buttons and from each other
    temp_slider_y = playback_y + button_height + 30  # First slider position
    tokens_slider_y = temp_slider_y + 60  # Second slider position with good vertical spacing
    
    # Create properly centered sliders
    temp_slider = Slider(
        slider_x, 
        temp_slider_y,
        slider_width, 
        10, 
        0.5, 1.5, 0.9, 
        "Temperature",
        integer_only=False  # Explicitly set to false for clarity
    )

    tokens_slider = Slider(
        slider_x, 
        tokens_slider_y,
        slider_width, 
        10, 
        64, 512, 256, 
        "Max Tokens",
        integer_only=True  # Set to true to ensure integers only
    )
    
    # Add all AI controls to the panel
    ai_panel.add_element(generate_button)
    ai_panel.add_element(accept_button)
    ai_panel.add_element(retry_button)
    ai_panel.add_element(play_button)
    ai_panel.add_element(pause_button)
    
    # Combine all buttons for the main list
    buttons = [
        record_button, stop_button, save_midi_button,  # Recording controls
        generate_button, accept_button, retry_button,  # AI composition controls
        play_button, pause_button                      # Playback controls
    ]
    
    # Create status message handler
    status = StatusMessage()
    
    # Draw keyboard
    if keyboard:
        keyboard.draw(screen)
    
    # Draw UI panels with all buttons
    panel.draw(screen)
    ai_panel.draw(screen)
    
    # Draw sliders
    temp_slider.draw(screen)
    tokens_slider.draw(screen)
    
    pygame.display.update()
    return screen, keyboard, buttons, panel, ai_panel, status, temp_slider, tokens_slider

def play_until_user_exits(
    keys: List[kl.Key],
    key_sounds: List[pygame.mixer.Sound],
    keyboard: klp.KeyboardLayout,
    screen: pygame.Surface,
    buttons: List[Button],
    framerate_hz: int,
    channels: int,
    tones: List[int],
    anchor_note: Optional[str],
    panel: UIPanel = None,
    ai_panel: UIPanel = None,
    status: StatusMessage = None,
    temp_slider = None,
    tokens_slider = None,
):
    sound_by_key = dict(zip(keys, key_sounds))
    playing = True
    
    # Initialize the recorder
    recorder = NoteRecorder(framerate_hz, channels, tones)
    
    # Initialize MIDI player
    midi_player = MIDIPlayer(status_callback=status.set_message if status else None)
    
    # Create recording indicator animation variables
    recording_indicator_size = 10
    recording_indicator_alpha = 255
    recording_indicator_growing = False
    
    # Track AI generation state
    ai_generation_active = False
    
    # Frame timing for smoother animations
    clock = pygame.time.Clock()
    
    while playing:
        mouse_pos = pygame.mouse.get_pos()
        
        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)
        
        # Update slider hover states
        if temp_slider:
            temp_slider.update(mouse_pos)
        if tokens_slider:
            tokens_slider.update(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False
                break
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    playing = False
                    break
                    
                key = keyboard.get_key(event)
                if key is None:
                    continue
                try:
                    key_index = keys.index(key)
                    sound = sound_by_key[key]
                except (KeyError, ValueError):
                    continue
                
                sound.stop()
                sound.play(fade_ms=SOUND_FADE_MILLISECONDS)
                
                # Record note if recording
                recorder.note_down(key, sound, key_index)
                
            elif event.type == pygame.KEYUP:
                key = keyboard.get_key(event)
                if key is None:
                    continue
                try:
                    sound = sound_by_key[key]
                except KeyError:
                    continue
                
                sound.fadeout(SOUND_FADE_MILLISECONDS)
                
                # Record note release if recording
                recorder.note_up(key)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Process slider events first - they take priority
                slider_handled = False
                
                # Check if either slider is clicked
                if temp_slider:
                    slider_handled = temp_slider.handle_event(event)
                
                if not slider_handled and tokens_slider:
                    slider_handled = tokens_slider.handle_event(event)
                
                # Only process button clicks if we didn't interact with sliders
                if not slider_handled:
                    # Check for recording button clicks
                    if buttons[0].is_clicked(pos):  # Record button
                        recorder.start_recording()
                        if status:
                            status.set_message("Recording started...", (255, 100, 100))
                        
                        # Update button active states
                        buttons[0].is_active = True
                        buttons[1].is_active = False
                        
                    elif buttons[1].is_clicked(pos):  # Stop button
                        recorder.stop_recording()
                        if status:
                            status.set_message("Recording stopped. Ready to save or generate.", (100, 255, 100))
                        
                        # Update button active states
                        buttons[0].is_active = False
                        buttons[1].is_active = True
                        
                    elif buttons[2].is_clicked(pos):  # Save MIDI button
                        if recorder.save_midi_recording(anchor_note):
                            if status:
                                status.set_message(f"MIDI recording saved to {RECORDINGS_FOLDER}", (100, 100, 255))
                        else:
                            if status:
                                status.set_message("No recording to save", (255, 100, 100))
                    
                    # Check for AI button clicks
                    elif buttons[3].is_clicked(pos):  # Generate button
                        if not recorder.midi_notes:
                            if status:
                                status.set_message("Record something first!", (255, 100, 100))
                        else:
                            if status:
                                status.set_message("Generating AI continuation...", (100, 100, 255))
                            
                            # Show active state
                            buttons[3].is_active = True
                            ai_generation_active = True
                            
                            # Use parameters from sliders
                            temperature = temp_slider.value if temp_slider else 0.9
                            max_tokens = int(tokens_slider.value) if tokens_slider else 256
                            
                            try:
                                # Run in a separate thread to avoid blocking UI
                                import threading
                                
                                def generate_task():
                                    # Save current recording to temporary MIDI
                                    temp_midi_path = os.path.join(recorder.recordings_dir, "temp_recording.mid")
                                    recorder.save_midi_recording(output_path=temp_midi_path, anchor_note=anchor_note)
                                    
                                    # Initialize AI if not already
                                    if not hasattr(recorder, 'ai_composer') or recorder.ai_composer is None:
                                        from ai_composer import AIComposer
                                        recorder.ai_composer = AIComposer()
                                    
                                    # Process recording and generate continuation
                                    encoded_input = recorder.ai_composer.process_user_recording(temp_midi_path)
                                    
                                    generated_output = recorder.ai_composer.generate_continuation(
                                        encoded_input, temperature=temperature, 
                                        top_k=100, max_new_tokens=max_tokens
                                    )
                                    
                                    # Store results
                                    recorder.current_continuation = generated_output
                                    
                                    # Create combined MIDI
                                    combined_path = os.path.join(recorder.recordings_dir, "ai_continuation.mid")
                                    recorder.ai_composer.create_combined_midi(
                                        encoded_input, generated_output, combined_path
                                    )
                                    
                                    # Update UI from main thread
                                    nonlocal ai_generation_active
                                    ai_generation_active = False
                                    buttons[3].is_active = False
                                    
                                    if status:
                                        status.set_message("AI continuation ready! Press PLAY to listen.", (100, 255, 100))
                                    
                                    # Load for playback
                                    midi_player.load_midi(combined_path)
                                    
                                # Start generation thread
                                threading.Thread(target=generate_task).start()
                                
                            except Exception as e:
                                ai_generation_active = False
                                buttons[3].is_active = False
                                if status:
                                    status.set_message(f"Error generating: {str(e)}", (255, 100, 100))
                    
                    elif buttons[4].is_clicked(pos):  # Accept button
                        if not hasattr(recorder, 'current_continuation') or not recorder.current_continuation:
                            if status:
                                status.set_message("Generate a continuation first!", (255, 100, 100))
                        else:
                            try:
                                # Save combined MIDI as final composition
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                final_path = os.path.join(recorder.recordings_dir, f"composition_{timestamp}.mid")
                                
                                # Copy the current combined file (or save it again)
                                combined_path = os.path.join(recorder.recordings_dir, "ai_continuation.mid")
                                shutil.copy(combined_path, final_path)
                                
                                if status:
                                    status.set_message(f"Composition saved to {os.path.basename(final_path)}", (100, 255, 100))
                            except Exception as e:
                                if status:
                                    status.set_message(f"Error saving: {str(e)}", (255, 100, 100))
                    
                    elif buttons[5].is_clicked(pos):  # Retry button
                        if not recorder.midi_notes:
                            if status:
                                status.set_message("Record something first!", (255, 100, 100))
                        else:
                            # Similar to generate but with different parameters for variety
                            if status:
                                status.set_message("Generating new AI continuation...", (220, 160, 40))
                            
                            # Show active state
                            buttons[5].is_active = True
                            ai_generation_active = True
                            
                            # Use slightly higher temperature for more variation
                            temperature = min(1.2, temp_slider.value + 0.1) if temp_slider else 1.0
                            max_tokens = int(tokens_slider.value) if tokens_slider else 256
                            
                            try:
                                # Similar generation logic as the generate button
                                import threading
                                
                                def retry_task():
                                    # Save current recording to temporary MIDI
                                    temp_midi_path = os.path.join(recorder.recordings_dir, "temp_recording.mid")
                                    recorder.save_midi_recording(output_path=temp_midi_path, anchor_note=anchor_note)
                                    
                                    # Initialize AI if not already
                                    if not hasattr(recorder, 'ai_composer') or recorder.ai_composer is None:
                                        from ai_composer import AIComposer
                                        recorder.ai_composer = AIComposer()
                                    
                                    # Process recording and generate continuation
                                    encoded_input = recorder.ai_composer.process_user_recording(temp_midi_path)
                                    
                                    generated_output = recorder.ai_composer.generate_continuation(
                                        encoded_input, temperature=temperature, 
                                        top_k=150,  # Higher top_k for more diversity
                                        max_new_tokens=max_tokens
                                    )
                                    
                                    # Store results
                                    recorder.current_continuation = generated_output
                                    
                                    # Create combined MIDI
                                    combined_path = os.path.join(recorder.recordings_dir, "ai_continuation.mid")
                                    recorder.ai_composer.create_combined_midi(
                                        encoded_input, generated_output, combined_path
                                    )
                                    
                                    # Update UI from main thread
                                    nonlocal ai_generation_active
                                    ai_generation_active = False
                                    buttons[5].is_active = False
                                    
                                    if status:
                                        status.set_message("New AI continuation ready! Press PLAY to listen.", (220, 160, 40))
                                    
                                    # Load for playback
                                    midi_player.load_midi(combined_path)
                                    
                                # Start generation thread
                                threading.Thread(target=retry_task).start()
                                
                            except Exception as e:
                                ai_generation_active = False
                                buttons[5].is_active = False
                                if status:
                                    status.set_message(f"Error generating: {str(e)}", (255, 100, 100))
                    
                    # Check playback controls
                    elif buttons[6].is_clicked(pos):  # Play button
                        midi_player.play()
                    
                    elif buttons[7].is_clicked(pos):  # Pause button
                        midi_player.pause()
                
            # Handle slider motion events
            elif event.type == pygame.MOUSEMOTION:
                # Update sliders during mouse motion for smoother dragging experience
                if temp_slider:
                    temp_slider.handle_event(event)
                if tokens_slider:
                    tokens_slider.handle_event(event)
            
            # For MOUSEBUTTONUP events:
            elif event.type == pygame.MOUSEBUTTONUP:
                # Make sure to update slider dragging state
                if temp_slider:
                    temp_slider.handle_event(event)
                if tokens_slider:
                    tokens_slider.handle_event(event)

            # For MOUSEMOTION events:
            elif event.type == pygame.MOUSEMOTION:
                # Update slider dragging and hovering
                if temp_slider:
                    temp_slider.handle_event(event)
                if tokens_slider:
                    tokens_slider.handle_event(event)
        
        # Redraw screen
        screen.fill((20, 20, 25))  # Dark blue-gray background
        
        # Draw keyboard
        keyboard.draw(screen)
        
        # Draw UI panels with all buttons
        if panel:
            panel.draw(screen)
        
        if ai_panel:
            ai_panel.draw(screen)
        
        # Draw sliders
        if temp_slider:
            temp_slider.draw(screen)
        
        if tokens_slider:
            tokens_slider.draw(screen)
        
        # Display status message
        if status:
            status_x = (screen.get_width() - 300) // 2  # Center the status message
            status_y = ai_panel.rect.y + ai_panel.rect.height - 40  # Position at bottom of AI panel
            status.draw(screen, (status_x, status_y))
        
        # Update and draw recording indicator if recording
        if recorder.recording:
            # Animate the recording indicator
            if recording_indicator_growing:
                recording_indicator_size += 0.5
                if recording_indicator_size >= 14:
                    recording_indicator_growing = False
            else:
                recording_indicator_size -= 0.5
                if recording_indicator_size <= 8:
                    recording_indicator_growing = True
            
            # Draw animated recording indicator
            indicator_x = screen.get_width() - 40
            indicator_y = keyboard.rect.height + 40
            
            # Pulsating transparent circle behind
            pulse_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                pulse_surface, 
                15, 
                15, 
                int(recording_indicator_size * 1.5),
                (255, 0, 0, max(0, int(80 - (recording_indicator_size * 5))))
            )
            screen.blit(pulse_surface, (indicator_x - 15, indicator_y - 15))
            
            # Solid circle
            pygame.gfxdraw.filled_circle(screen, indicator_x, indicator_y, 
                                       int(recording_indicator_size), (255, 0, 0))
            pygame.gfxdraw.aacircle(screen, indicator_x, indicator_y, 
                                   int(recording_indicator_size), (255, 0, 0))
            
            # "REC" text
            rec_font = pygame.font.SysFont("Arial", 16, bold=True)
            rec_text = rec_font.render("REC", True, (255, 255, 255))
            screen.blit(rec_text, (indicator_x - 40, indicator_y - 8))
            
        # Draw AI generation loading indicator if active
        if ai_generation_active:
            # Draw loading spinner in the center of the AI panel
            spinner_x = screen.get_width() // 2
            spinner_y = ai_panel.rect.y + (ai_panel.rect.height // 2)
            
            # Spinning animation
            current_time = pygame.time.get_ticks()
            angle = (current_time % 1000) / 1000 * 360
            
            spinner_radius = 20
            for i in range(8):
                dot_angle = angle + i * 45
                dot_x = spinner_x + int(math.cos(math.radians(dot_angle)) * spinner_radius)
                dot_y = spinner_y + int(math.sin(math.radians(dot_angle)) * spinner_radius)
                
                # Fade out dots based on position in spin
                alpha = 255 - ((i * 25) % 255)
                pygame.gfxdraw.filled_circle(screen, dot_x, dot_y, 4, (100, 180, 255, alpha))
            
            # "Generating..." text
            gen_font = pygame.font.SysFont("Arial", 16, bold=True)
            gen_text = gen_font.render("Generating AI Continuation...", True, (200, 200, 255))
            screen.blit(gen_text, (spinner_x - gen_text.get_width() // 2, spinner_y + 30))
            
        pygame.display.update()
        
        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
    print("Goodbye")
class NoteRecorder:
    def __init__(self, framerate_hz, channels, tones):
        self.recording = False
        self.notes = []  # List of (key, sound, start_time, duration)
        self.midi_notes = []  # List of (tone, start_time, duration)
        self.current_notes = {}  # Maps key to (sound, start_time)
        self.current_midi_notes = {}  # Maps key to (tone, start_time)
        self.framerate_hz = framerate_hz
        self.channels = channels
        self.start_time = 0
        self.tones = tones  # Store the tone values for MIDI conversion
        self.ai_composer = None  # Will be initialized on demand
        self.current_continuation = None
        self.combined_midi_path = None
        # Ensure recordings directory exists
        self.recordings_dir = os.path.join(CURRENT_WORKING_DIR, RECORDINGS_FOLDER)
        os.makedirs(self.recordings_dir, exist_ok=True)

    def initialize_ai(self):
        """Lazy initialization of AI composer"""
        if self.ai_composer is None:
            from pianoputer.ai_composer import AIComposer
            self.ai_composer = AIComposer()
            
    def generate_continuation(self, temperature=0.9, top_k=100, max_tokens=256):
        """Generate AI continuation for current recording"""
        self.initialize_ai()
        
        # Save current recording to temporary MIDI
        temp_midi_path = os.path.join(self.recordings_dir, "temp_recording.mid")
        self.save_midi_recording(output_path=temp_midi_path)
        
        # Process recording for model input
        encoded_input = self.ai_composer.process_user_recording(temp_midi_path)
        
        # Generate continuation
        generated_output = self.ai_composer.generate_continuation(
            encoded_input, temperature, top_k, max_tokens)
        
        # Save the continuation
        self.current_continuation = generated_output
        
        # Create combined MIDI
        self.combined_midi_path = os.path.join(self.recordings_dir, "combined_recording.mid")
        self.ai_composer.create_combined_midi(
            encoded_input, generated_output, self.combined_midi_path)
        
        return self.combined_midi_path
        
    def accept_continuation(self):
        """Accept and save the current AI continuation"""
        if self.current_continuation:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(self.recordings_dir, f"composition_{timestamp}.mid")
            shutil.copy(self.combined_midi_path, final_path)
            return final_path
        return None
    
    def start_recording(self):
        self.recording = True
        self.notes = []
        self.midi_notes = []
        self.current_notes = {}
        self.current_midi_notes = {}
        self.start_time = time.time()
        print("Recording started...")
    
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        # Stop any notes that are still playing
        current_time = time.time()
        for key, (sound, start_time) in self.current_notes.items():
            duration = current_time - start_time
            self.notes.append((key, sound, start_time - self.start_time, duration))
        
        for key, (tone, start_time) in self.current_midi_notes.items():
            duration = current_time - start_time
            self.midi_notes.append((tone, start_time - self.start_time, duration))
        
        self.current_notes = {}
        self.current_midi_notes = {}
        print("Recording stopped.")
    
    def note_down(self, key, sound, key_index):
        if not self.recording:
            return
            
        current_time = time.time()
        self.current_notes[key] = (sound, current_time)
        
        # Also record for MIDI
        tone = self.tones[key_index]
        self.current_midi_notes[key] = (tone, current_time)
    
    def note_up(self, key):
        if not self.recording:
            return
            
        if key in self.current_notes:
            sound, start_time = self.current_notes.pop(key)
            current_time = time.time()
            duration = current_time - start_time
            self.notes.append((key, sound, start_time - self.start_time, duration))
        
        if key in self.current_midi_notes:
            tone, start_time = self.current_midi_notes.pop(key)
            current_time = time.time()
            duration = current_time - start_time
            self.midi_notes.append((tone, start_time - self.start_time, duration))
    
    def save_wav_recording(self):
        if not self.notes:
            print("No recording to save.")
            return False
            
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pianoputer_recording_{timestamp}.wav"
        filepath = os.path.join(self.recordings_dir, filename)
        
        # Mix all the notes into a single audio file
        # First, determine the total length of the recording
        end_time = max(start_time + duration for _, _, start_time, duration in self.notes)
        
        # Create an empty audio array
        total_samples = int(end_time * self.framerate_hz)
        if self.channels == 1:
            mixed_audio = numpy.zeros(total_samples)
        else:
            mixed_audio = numpy.zeros((total_samples, self.channels))
        
        # Mix in each note
        for _, sound, start_time, duration in self.notes:
            sound_array = pygame.sndarray.samples(sound)
            start_sample = int(start_time * self.framerate_hz)
            end_sample = min(start_sample + len(sound_array), total_samples)
            
            # Only use as much of the sound as fits within the duration
            sound_end = min(len(sound_array), int(duration * self.framerate_hz))
            
            # Mix in the sound
            if start_sample < total_samples:
                if self.channels == 1:
                    mixed_audio[start_sample:end_sample] += sound_array[:sound_end]
                else:
                    # Handle multichannel audio
                    samples_to_mix = min(end_sample - start_sample, sound_end)
                    mixed_audio[start_sample:start_sample + samples_to_mix] += sound_array[:samples_to_mix]
        
        # Normalize to avoid clipping
        if numpy.max(numpy.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / numpy.max(numpy.abs(mixed_audio))
        
        # Save the mixed audio
        soundfile.write(filepath, mixed_audio, self.framerate_hz, DESCRIPTOR_32BIT)
        print(f"WAV recording saved to: {filepath}")
        return True
    
    def save_midi_recording(self, anchor_note=None):
        if not self.midi_notes:
            print("No MIDI notes to save.")
            return False
        
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pianoputer_midi_{timestamp}.mid"
        filepath = os.path.join(self.recordings_dir, filename)
        
        # Create a new MIDI file with one track
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=MIDI_TEMPO, time=0))
        
        # Determine the reference note
        if anchor_note and anchor_note in LETTER_KEYS_TO_INDEX:
            # If we have an anchor note like 'c', calculate the base MIDI note
            reference_midi_note = MIDI_C4 - LETTER_KEYS_TO_INDEX[anchor_note]
        else:
            # Default to middle C
            reference_midi_note = MIDI_C4
        
        # Sort notes by start time
        sorted_notes = sorted(self.midi_notes, key=lambda x: x[1])
        
        # Convert to MIDI events
        last_time = 0
        for tone, start_time, duration in sorted_notes:
            # Convert real time to MIDI ticks (assuming 480 ticks per beat)
            start_ticks = int(start_time * 1000)  # Convert to milliseconds
            duration_ticks = int(duration * 1000)  # Convert to milliseconds
            
            # Calculate delta time (time since last event)
            delta_start = start_ticks - last_time
            
            # Calculate MIDI note number from semitone shift
            midi_note = reference_midi_note + tone
            
            # Ensure note is in valid MIDI range (0-127)
            if 0 <= midi_note <= 127:
                # Add note on message
                track.append(mido.Message('note_on', note=midi_note, velocity=MIDI_VELOCITY, time=delta_start))
                
                # Add note off message
                track.append(mido.Message('note_off', note=midi_note, velocity=0, time=duration_ticks))
                
                # Update last time
                last_time = start_ticks + duration_ticks
        
        # Save the MIDI file
        mid.save(filepath)
        print(f"MIDI recording saved to: {filepath}")
        return True

def get_audio_data(wav_path: str) -> Tuple:
    audio_data, framerate_hz = soundfile.read(wav_path)
    array_shape = audio_data.shape
    if len(array_shape) == 1:
        channels = 1
    else:
        channels = array_shape[1]
    return audio_data, framerate_hz, channels


def process_args(parser: argparse.ArgumentParser, args: Optional[List]) -> Tuple:
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Enable warnings from scipy if requested
    if not args.verbose:
        warnings.simplefilter("ignore")

    wav_path = args.wav
    if wav_path.startswith(AUDIO_ASSET_PREFIX):
        wav_path = os.path.join(CURRENT_WORKING_DIR, wav_path)

    keyboard_path = args.keyboard
    if keyboard_path.startswith(KEYBOARD_ASSET_PREFIX):
        keyboard_path = os.path.join(CURRENT_WORKING_DIR, keyboard_path)
    return wav_path, keyboard_path, args.clear_cache


def play_pianoputer(args: Optional[List[str]] = None):
    parser = get_parser()
    wav_path, keyboard_path, clear_cache = process_args(parser, args)
    audio_data, framerate_hz, channels = get_audio_data(wav_path)
    results = get_keyboard_info(keyboard_path)
    keys, tones, color_to_key, key_color, key_txt_color, anchor_note = results
    key_sounds = get_or_create_key_sounds(
        wav_path, framerate_hz, channels, tones, clear_cache, keys
    )

    # Unpack all 8 return values
    screen, keyboard, buttons, panel, ai_panel, status, temp_slider, tokens_slider = configure_pygame_audio_and_set_ui(
        framerate_hz, channels, keyboard_path, color_to_key, key_color, key_txt_color
    )
    
    print(
        "Ready for you to play!\n"
        "Press the keys on your keyboard. Use the buttons to record your performance.\n"
        "You can save as WAV audio or MIDI file.\n"
        "To exit press ESC or close the pygame window"
    )
    
    # Pass all parameters to play_until_user_exits
    play_until_user_exits(keys, list(key_sounds), keyboard, screen, buttons, 
                          framerate_hz, channels, tones, anchor_note, panel, ai_panel, status,
                          temp_slider, tokens_slider)


if __name__ == "__main__":
    play_pianoputer()