#!/usr/bin/env python
import matplotlib.pyplot as plt
import pretty_midi
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
    def __init__(self, x, y, width, height, color, text, icon=None, text_color=(255, 255, 255, 255), 
                 font_size=16, border_radius=12, icon_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.text_color = text_color
        self.icon = icon
        self.is_active = False
        self.is_hovering = False
        
        # Modern design properties
        self.border_radius = border_radius
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.font = pygame.font.SysFont(self.font_name, font_size, bold=False)
        self.icon_size = icon_size
        self.shadow_offset = 2
        self.shadow_alpha = 120
        self.glow_alpha = 0
        self.animation_speed = 0.2
        self.animation_progress = 0
        
        # Calculate different color variations for states
        self.hover_color = self._get_hover_color(color)
        self.active_color = self._get_active_color(color)
        self.current_color = self.color
        
    def _get_hover_color(self, color):
        """Return a slightly lighter version of the color for hover state"""
        r, g, b = color[:3]
        factor = 1.15
        return (min(int(r * factor), 255), 
                min(int(g * factor), 255), 
                min(int(b * factor), 255), 
                color[3] if len(color) > 3 else 255)
    
    def _get_active_color(self, color):
        """Return a slightly darker version of the color for active state"""
        r, g, b = color[:3]
        factor = 0.85
        return (int(r * factor), 
                int(g * factor), 
                int(b * factor), 
                color[3] if len(color) > 3 else 255)
    
    def update(self, mouse_pos, dt=1/60):
        """Update button state and animations based on mouse position"""
        prev_hovering = self.is_hovering
        self.is_hovering = self.rect.collidepoint(mouse_pos)
        
        # Animate color changes for smoother transitions
        target_color = self.active_color if self.is_active else (
                    self.hover_color if self.is_hovering else self.color)
        
        # Animate glow effect when hovering begins
        if not prev_hovering and self.is_hovering:
            self.glow_alpha = 60  # Start glow effect
        
        # Fade out glow effect
        if self.glow_alpha > 0:
            self.glow_alpha = max(0, self.glow_alpha - 120 * dt)
            
        # Smooth color transition
        self.current_color = self._interpolate_color(self.current_color, target_color, self.animation_speed)
        
        # Update pulse animation for recording button
        if self.icon == "record" and self.is_active:
            self.animation_progress = (self.animation_progress + 3 * dt) % 1.0
    
    def _interpolate_color(self, color1, color2, fraction):
        """Smoothly interpolate between two colors"""
        r1, g1, b1 = color1[:3]
        r2, g2, b2 = color2[:3]
        
        r = int(r1 + (r2 - r1) * fraction)
        g = int(g1 + (g2 - g1) * fraction)
        b = int(b1 + (b2 - b1) * fraction)
        
        alpha = color1[3] if len(color1) > 3 else 255
        return (r, g, b, alpha)
    
    def draw(self, screen):
        # Draw subtle shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += self.shadow_offset
        shadow_rect.x += self.shadow_offset // 2
        self._draw_rounded_rect(screen, shadow_rect, (20, 20, 30, self.shadow_alpha), self.border_radius)
        
        # Draw glow effect when hovering (for extra polish)
        if self.glow_alpha > 0:
            glow_rect = self.rect.copy()
            glow_rect.inflate_ip(6, 6)
            self._draw_rounded_rect(screen, glow_rect, 
                                  (self.color[0], self.color[1], self.color[2], self.glow_alpha), 
                                  self.border_radius + 3)
        
        # Draw main button with current interpolated color
        self._draw_rounded_rect(screen, self.rect, self.current_color, self.border_radius)
        
        # Calculate text and icon positions
        if self.icon:
            # If there's an icon, position both icon and text
            icon_padding = 10
            
            if len(self.text) > 0:
                # Show both icon and text
                text_surface = self.font.render(self.text, True, self.text_color)
                total_width = self.icon_size + icon_padding + text_surface.get_width()
                
                icon_x = self.rect.centerx - total_width // 2
                icon_y = self.rect.centery - self.icon_size // 2
                
                # Draw icon
                self._draw_icon(screen, icon_x, icon_y, self.icon_size)
                
                # Draw text
                text_x = icon_x + self.icon_size + icon_padding
                text_y = self.rect.centery - text_surface.get_height() // 2
                screen.blit(text_surface, (text_x, text_y))
            else:
                # Icon only button (centered)
                icon_x = self.rect.centerx - self.icon_size // 2
                icon_y = self.rect.centery - self.icon_size // 2
                self._draw_icon(screen, icon_x, icon_y, self.icon_size)
        else:
            # Text only button
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
    
    def _draw_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners using pygame.gfxdraw for anti-aliasing"""
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
    
    def _draw_icon(self, screen, x, y, size):
        """Draw modern styled icons based on type"""
        if self.icon == "record":
            # Modern recording icon - filled circle with pulsating ring
            color = (255, 80, 80)  # Brighter red for modern look
            center_x, center_y = x + size//2, y + size//2
            
            # Add pulsating animation when active
            if self.is_active:
                # Use animation_progress for smooth pulsating
                pulse_factor = 0.5 + 0.5 * abs(math.sin(self.animation_progress * math.pi * 2))
                
                # Draw pulsating outer glow
                outer_size = int((size//2) * (1 + pulse_factor * 0.5))
                for r in range(outer_size, size//2 - 4, -1):
                    alpha = int(100 * (1 - (r - size//2 + 4) / (outer_size - size//2 + 4)))
                    pygame.gfxdraw.aacircle(screen, center_x, center_y, r, (255, 0, 0, alpha))
            
            # Inner filled circle
            pygame.gfxdraw.filled_circle(screen, center_x, center_y, size//2 - 4, color)
            pygame.gfxdraw.aacircle(screen, center_x, center_y, size//2 - 4, color)
        
        elif self.icon == "stop":
            # Modern stop icon - rounded square
            color = (255, 255, 255)
            stop_rect = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
            pygame.draw.rect(screen, color, stop_rect, border_radius=2)
        
        elif self.icon == "play":
            # Modern play triangle
            color = (120, 220, 120)
            points = [
                (x + 4, y + 2),
                (x + 4, y + size - 2),
                (x + size - 2, y + size//2)
            ]
            pygame.gfxdraw.filled_polygon(screen, points, color)
            pygame.gfxdraw.aapolygon(screen, points, color)
        
        elif self.icon == "pause":
            # Modern pause icon - two rounded bars
            color = (255, 255, 255)
            bar_width = (size - 12) // 2
            pygame.draw.rect(screen, color, (x + 4, y + 4, bar_width, size - 8), border_radius=2)
            pygame.draw.rect(screen, color, (x + 8 + bar_width, y + 4, bar_width, size - 8), border_radius=2)
        
        elif self.icon == "save_midi":
            # Modern MIDI icon - simplified piano keys
            # White keys
            keys_color = (255, 255, 255)
            key_width = size // 4
            white_key_height = size - 6
            
            for i in range(3):
                pygame.draw.rect(screen, keys_color, 
                                (x + i * key_width + 3, y + 3, 
                                 key_width - 1, white_key_height),
                                border_radius=2)
            
            # Black keys
            black_key_height = white_key_height * 0.6
            pygame.draw.rect(screen, (40, 40, 40), 
                            (x + key_width * 0.7, y + 3, 
                             key_width * 0.6, black_key_height),
                            border_radius=1)
            pygame.draw.rect(screen, (40, 40, 40), 
                            (x + key_width * 1.7, y + 3, 
                             key_width * 0.6, black_key_height),
                            border_radius=1)
        
        elif self.icon == "generate":
            # AI generation icon (stylized sparkle/brain)
            color = (130, 180, 255)
            
            # Draw a stylized sparkle/star
            center_x, center_y = x + size//2, y + size//2
            outer_radius = size//2 - 2
            inner_radius = outer_radius // 2
            
            points = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                radius = outer_radius if i % 2 == 0 else inner_radius
                points.append((
                    center_x + int(radius * math.cos(angle)),
                    center_y + int(radius * math.sin(angle))
                ))
            
            pygame.gfxdraw.filled_polygon(screen, points, color)
            pygame.gfxdraw.aapolygon(screen, points, color)
        
        elif self.icon == "accept":
            # Checkmark icon
            color = (100, 220, 100)
            
            # Draw checkmark
            points = [
                (x + 4, y + size//2),
                (x + size//3, y + size - 6),
                (x + size - 4, y + 4)
            ]
            
            # Draw with anti-aliasing
            pygame.gfxdraw.aapolygon(screen, points, color)
            
            # Draw lines with thickness
            pygame.draw.lines(screen, color, False, points, 2)
        
        elif self.icon == "retry":
            # Retry/refresh icon
            color = (220, 180, 40)
            
            # Draw circular arrow
            center_x, center_y = x + size//2, y + size//2
            radius = size//2 - 4
            
            # Arc positions (in radians)
            start_angle = math.pi * 0.1
            end_angle = math.pi * 1.9
            
            # Draw the arc
            points = []
            for i in range(20):
                angle = start_angle + (end_angle - start_angle) * (i / 19)
                points.append((
                    center_x + int(radius * math.cos(angle)),
                    center_y + int(radius * math.sin(angle))
                ))
            
            # Draw the arc with anti-aliasing
            pygame.draw.lines(screen, color, False, points, 2)
            
            # Arrow head
            arrow_size = 4
            pygame.draw.polygon(screen, color, [
                (points[-1][0], points[-1][1]),
                (points[-1][0] - arrow_size, points[-1][1] - arrow_size),
                (points[-1][0] + arrow_size, points[-1][1] - arrow_size)
            ])
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def set_active(self, active):
        self.is_active = active

class UIPanel:
    def __init__(self, x, y, width, height, bg_color=(30, 32, 36), 
                 border_color=None, title=None, title_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.elements = []
        self.title = title
        self.title_color = title_color
        
        # Calculate border color from background if not specified
        if border_color is None:
            # Make border slightly lighter than background
            r, g, b = bg_color[:3]
            factor = 1.3
            self.border_color = (min(int(r * factor), 255),
                                min(int(g * factor), 255),
                                min(int(b * factor), 255),
                                100)  # Semi-transparent
        else:
            self.border_color = border_color
            
        # Modern UI properties
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.title_font = pygame.font.SysFont(self.font_name, 18, bold=True)
        self.border_radius = 12
        self.shadow_size = 5
        self.shadow_alpha = 80
        
        # If there's a title, add padding at the top for it
        self.content_rect = self.rect.copy()
        if self.title:
            title_height = 30
            self.content_rect.y += title_height
            self.content_rect.height -= title_height
    
    def set_title(self, title, color=(255, 255, 255)):
        self.title = title
        self.title_color = color
        
    def add_element(self, element):
        self.elements.append(element)
        
    def draw(self, screen):
        # Draw shadow first
        shadow_rect = self.rect.copy()
        shadow_rect.inflate_ip(4, 4)
        shadow_rect.move_ip(2, 2)
        self._draw_rounded_rect(screen, shadow_rect, (10, 10, 15, self.shadow_alpha), self.border_radius)
        
        # Draw panel background with rounded corners
        self._draw_rounded_rect(screen, self.rect, self.bg_color, self.border_radius)
        
        # Draw subtle gradient overlay for depth (top lighter, bottom darker)
        gradient_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        
        # Create subtle gradient from top to bottom
        for i in range(self.rect.height):
            alpha = 10 - int(20 * (i / self.rect.height))  # Fade from light to dark
            if alpha > 0:
                pygame.draw.line(gradient_surface, (255, 255, 255, alpha), 
                             (0, i), (self.rect.width, i))
            else:
                pygame.draw.line(gradient_surface, (0, 0, 0, -alpha), 
                             (0, i), (self.rect.width, i))
        
        # Draw the gradient with rounded corners
        self._draw_rounded_surface(screen, gradient_surface, self.rect.topleft, self.border_radius)
        
        # Draw panel border with rounded corners
        pygame.draw.rect(screen, self.border_color, self.rect, 1, border_radius=self.border_radius)
        
        # Draw title if set
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            
            # Create background for title area
            title_height = 30
            title_bg_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, title_height)
            
            # Draw rounded corners only at top
            self._draw_top_rounded_rect(screen, title_bg_rect, 
                                     (self.bg_color[0], self.bg_color[1], self.bg_color[2], 200), 
                                     self.border_radius)
            
            # Position title centered
            title_rect = title_surface.get_rect(
                midtop=(self.rect.centerx, self.rect.y + 8)
            )
            screen.blit(title_surface, title_rect)
            
            # Draw subtle separator line
            separator_y = self.rect.y + title_height
            pygame.draw.line(screen, self.border_color, 
                          (self.rect.x + 10, separator_y), 
                          (self.rect.right - 10, separator_y))
        
        # Draw all contained elements
        for element in self.elements:
            element.draw(screen)
    
    def _draw_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners using pygame.gfxdraw for anti-aliasing"""
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
    
    def _draw_top_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners only at the top"""
        if radius <= 0:
            pygame.draw.rect(surface, color, rect)
            return
            
        # Get the rectangle dimensions
        x, y, width, height = rect
        
        # Draw the rectangle with rounded corners only at top
        pygame.gfxdraw.box(surface, (x + radius, y, width - 2 * radius, height), color)
        pygame.gfxdraw.box(surface, (x, y + radius, width, height - radius), color)
        
        # Draw the two top rounded corners
        pygame.gfxdraw.filled_circle(surface, x + radius, y + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + width - radius - 1, y + radius, radius, color)
    
    def _draw_rounded_surface(self, target_surface, surface, pos, radius):
        """Draw a surface with rounded corners using a mask"""
        # Create a mask surface 
        mask = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        self._draw_rounded_rect(mask, mask.get_rect(), (255, 255, 255), radius)
        
        # Apply the mask
        masked_surface = surface.copy()
        masked_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Draw to target
        target_surface.blit(masked_surface, pos)
    
    def update(self, mouse_pos):
        """Update all elements in the panel"""
        for element in self.elements:
            if hasattr(element, 'update'):
                element.update(mouse_pos)
    
    def handle_event(self, event):
        """Handle events for all elements in the panel"""
        for element in self.elements:
            if hasattr(element, 'handle_event'):
                if element.handle_event(event):
                    return True
        return False

class StatusMessage:
    def __init__(self):
        self.message = ""
        self.color = (255, 255, 255)
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.font = pygame.font.SysFont(self.font_name, 16, bold=True)
        self.start_time = 0
        self.duration = 4  # seconds to display the message
        self.fade_duration = 0.8  # seconds to fade out
        self.background_color = (40, 42, 48, 230)  # Dark, semi-transparent background
        
        # Animation properties
        self.current_opacity = 0
        self.target_opacity = 0
        self.opacity_speed = 8  # Speed of fade in/out
        self.slide_offset = 0
        self.slide_target = 0
        self.slide_speed = 10  # Speed of slide animation
        
    def set_message(self, message, color=(255, 255, 255)):
        if message != self.message:
            self.slide_offset = 10  # Start slide-in animation
            self.slide_target = 0
            
        self.message = message
        self.color = color
        self.start_time = time.time()
        self.target_opacity = 255
        
    def update(self, dt=1/60):
        """Update animations"""
        if not self.message:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # If the message has expired, start fade out
        if elapsed > self.duration:
            self.target_opacity = 0
        
        # If completely faded out and expired, clear the message
        if self.current_opacity <= 0 and elapsed > self.duration:
            self.message = ""
            return
            
        # Animate opacity
        if self.current_opacity < self.target_opacity:
            self.current_opacity = min(self.target_opacity, self.current_opacity + self.opacity_speed * 255 * dt)
        elif self.current_opacity > self.target_opacity:
            self.current_opacity = max(self.target_opacity, self.current_opacity - self.opacity_speed * 255 * dt)
            
        # Animate slide
        if self.slide_offset > self.slide_target:
            self.slide_offset = max(self.slide_target, self.slide_offset - self.slide_speed * dt * 60)
        elif self.slide_offset < self.slide_target:
            self.slide_offset = min(self.slide_target, self.slide_offset + self.slide_speed * dt * 60)
        
    def draw(self, screen, position):
        if not self.message or self.current_opacity <= 0:
            return
        
        # Get message dimensions
        text_surface = self.font.render(self.message, True, self.color)
        text_rect = text_surface.get_rect()
        
        # Create a background surface with padding and rounded corners
        padding = 12
        bg_width = text_rect.width + padding * 2
        bg_height = text_rect.height + padding * 1.5
        
        bg_rect = pygame.Rect(
            position[0] - padding + self.slide_offset,
            position[1] - padding // 2,
            bg_width,
            bg_height
        )
        
        # Draw background with rounded corners
        rounded_rect_radius = 8
        
        # Background shadow for depth
        shadow_offset = 2
        shadow_rect = bg_rect.copy()
        shadow_rect.x += shadow_offset
        shadow_rect.y += shadow_offset
        self._draw_rounded_rect(
            screen, 
            shadow_rect, 
            (20, 20, 25, int(self.current_opacity * 0.5)), 
            rounded_rect_radius
        )
        
        # Main background
        bg_color = self.background_color[:3] + (int(self.current_opacity * 0.9),)
        self._draw_rounded_rect(screen, bg_rect, bg_color, rounded_rect_radius)
        
        # Add subtle gradient for depth
        gradient_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        for i in range(bg_rect.height):
            alpha = 5 - int(10 * (i / bg_rect.height))  # Fade from light to dark
            if alpha > 0:
                pygame.draw.line(gradient_surface, (255, 255, 255, alpha * (self.current_opacity/255)), 
                              (0, i), (bg_rect.width, i))
        
        # Apply rounded corners to gradient
        self._draw_rounded_surface(screen, gradient_surface, bg_rect.topleft, rounded_rect_radius)
        
        # Draw text with current opacity
        alpha_text = text_surface.copy()
        alpha_text.set_alpha(int(self.current_opacity))
        
        # Adjusted text position (centered in background)
        text_pos = (
            bg_rect.centerx - text_rect.width // 2,
            bg_rect.centery - text_rect.height // 2
        )
        
        screen.blit(alpha_text, text_pos)
        
        # Add subtle border
        border_color = (255, 255, 255, int(self.current_opacity * 0.2))
        pygame.draw.rect(screen, border_color, bg_rect, 1, border_radius=rounded_rect_radius)
    
    def _draw_rounded_rect(self, surface, rect, color, radius):
        """Draw a rectangle with rounded corners using pygame.gfxdraw"""
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
    
    def _draw_rounded_surface(self, target_surface, surface, pos, radius):
        """Draw a surface with rounded corners using a mask"""
        # Create a mask surface 
        mask = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        self._draw_rounded_rect(mask, mask.get_rect(), (255, 255, 255), radius)
        
        # Apply the mask
        masked_surface = surface.copy()
        masked_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Draw to target
        target_surface.blit(masked_surface, pos)

class MIDIPlayer:
    def __init__(self, status_callback=None):
        self.current_midi = None
        self.playing = False
        self.status_callback = status_callback
        self.process = None
        self.midi_directory = os.path.join(CURRENT_WORKING_DIR, RECORDINGS_FOLDER)
        # Ensure directory exists
        os.makedirs(self.midi_directory, exist_ok=True)
        
    def set_midi_directory(self, directory):
        """Set the directory to look for MIDI files"""
        if os.path.exists(directory) and os.path.isdir(directory):
            self.midi_directory = directory
            return True
        return False
        
    def load_midi(self, midi_path):
        """Load a MIDI file for playback"""
        if not os.path.exists(midi_path):
            if self.status_callback:
                self.status_callback(f"MIDI file not found: {midi_path}", (255, 100, 100))
            return False
            
        self.current_midi = midi_path
        if self.status_callback:
            self.status_callback(f"Loaded: {os.path.basename(midi_path)}", (100, 255, 100))
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
    
    def get_midi_files(self):
        """Return a list of MIDI files in the directory"""
        midi_files = []
        try:
            for file in os.listdir(self.midi_directory):
                if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                    full_path = os.path.join(self.midi_directory, file)
                    midi_files.append((file, full_path))
            return midi_files
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error reading MIDI directory: {str(e)}", (255, 100, 100))
            return []

class FileBrowser:
    def __init__(self, x, y, width, height, directory=RECORDINGS_FOLDER, 
                 file_extension=".mid", bg_color=(30, 32, 36), 
                 file_selected_callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.directory = os.path.join(CURRENT_WORKING_DIR, directory)
        self.file_extension = file_extension
        self.bg_color = bg_color
        self.border_color = (80, 82, 86, 100)  # Slightly lighter border
        self.file_selected_callback = file_selected_callback
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        # UI properties
        self.border_radius = 10
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.title_font = pygame.font.SysFont(self.font_name, 18, bold=True)
        self.file_font = pygame.font.SysFont(self.font_name, 16)
        self.info_font = pygame.font.SysFont(self.font_name, 14, italic=True)
        
        # Scrolling properties
        self.scroll_y = 0
        self.max_scroll = 0
        self.scroll_speed = 20
        self.item_height = 30
        self.visible_items = (height - 60) // self.item_height  # Account for header and footer
        
        # Selection properties
        self.selected_file = None
        self.selected_index = -1
        self.hover_index = -1
        
        # File list
        self.files = []
        self.refresh_file_list()
    
    def refresh_file_list(self):
        """Update the list of MIDI files in the directory"""
        self.files = []
        try:
            for file in os.listdir(self.directory):
                if file.lower().endswith(self.file_extension):
                    # Get file info
                    file_path = os.path.join(self.directory, file)
                    file_stats = os.stat(file_path)
                    # Store file info: (name, full path, modification time)
                    modified_time = datetime.datetime.fromtimestamp(file_stats.st_mtime)
                    modified_str = modified_time.strftime("%Y-%m-%d %H:%M")
                    self.files.append((file, file_path, modified_str))
            
            # Sort by modification time (newest first)
            self.files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            
            # Reset scroll position and update max scroll
            self.update_max_scroll()
            
        except Exception as e:
            print(f"Error refreshing file list: {e}")
    
    def update_max_scroll(self):
        """Update the maximum scroll value based on file list length"""
        total_items_height = len(self.files) * self.item_height
        visible_area_height = self.rect.height - 60  # Account for header and footer
        
        if total_items_height > visible_area_height:
            self.max_scroll = total_items_height - visible_area_height
        else:
            self.max_scroll = 0
            self.scroll_y = 0
    
    def draw(self, screen):
        """Draw the file browser"""
        # Draw background
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(screen, self.border_color, self.rect, width=1, border_radius=self.border_radius)
        
        # Draw title
        title_text = self.title_font.render("MIDI Files", True, (220, 220, 230))
        title_rect = title_text.get_rect(
            midtop=(self.rect.centerx, self.rect.y + 10)
        )
        screen.blit(title_text, title_rect)
        
        # Draw separator line
        pygame.draw.line(screen, self.border_color, 
                     (self.rect.x + 10, self.rect.y + 40), 
                     (self.rect.right - 10, self.rect.y + 40),
                     width=1)
        
        # Create a clipping rect for the file list
        list_rect = pygame.Rect(
            self.rect.x + 5,
            self.rect.y + 45,
            self.rect.width - 10,
            self.rect.height - 55
        )
        
        # Set clipping area
        original_clip = screen.get_clip()
        screen.set_clip(list_rect)
        
        # Draw file list
        if not self.files:
            # Draw "No files" message
            info_text = self.info_font.render("No MIDI files found", True, (180, 180, 180))
            info_rect = info_text.get_rect(
                center=(self.rect.centerx, self.rect.y + 70)
            )
            screen.blit(info_text, info_rect)
        else:
            # Draw files
            y_pos = self.rect.y + 45 - self.scroll_y
            for i, (file_name, file_path, modified_date) in enumerate(self.files):
                item_rect = pygame.Rect(
                    self.rect.x + 5,
                    y_pos,
                    self.rect.width - 10,
                    self.item_height
                )
                
                # Skip if completely outside visible area
                if y_pos + self.item_height < self.rect.y + 45 or y_pos > self.rect.y + self.rect.height - 10:
                    y_pos += self.item_height
                    continue
                
                # Determine item background color
                if i == self.selected_index:
                    # Selected item
                    bg_color = (70, 120, 200)
                elif i == self.hover_index:
                    # Hovered item
                    bg_color = (50, 52, 56)
                else:
                    # Normal item
                    bg_color = None
                
                # Draw item background
                if bg_color:
                    pygame.draw.rect(screen, bg_color, item_rect, border_radius=5)
                
                # Draw file name (truncate if too long)
                display_name = file_name
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."
                    
                name_color = (240, 240, 240) if i == self.selected_index else (220, 220, 220)
                name_text = self.file_font.render(display_name, True, name_color)
                screen.blit(name_text, (item_rect.x + 10, item_rect.y + 5))
                
                # Draw modified date
                date_color = (220, 220, 220) if i == self.selected_index else (160, 160, 160)
                date_text = self.info_font.render(modified_date, True, date_color)
                date_rect = date_text.get_rect(right=item_rect.right - 10, y=item_rect.y + 8)
                screen.blit(date_text, date_rect)
                
                y_pos += self.item_height
        
        # Reset clipping area
        screen.set_clip(original_clip)
        
        # Draw scroll indicators if necessary
        if self.max_scroll > 0:
            if self.scroll_y > 0:
                # Draw up arrow
                pygame.draw.polygon(screen, (180, 180, 180), [
                    (self.rect.right - 15, self.rect.y + 50),
                    (self.rect.right - 10, self.rect.y + 45),
                    (self.rect.right - 20, self.rect.y + 45)
                ])
            
            if self.scroll_y < self.max_scroll:
                # Draw down arrow
                pygame.draw.polygon(screen, (180, 180, 180), [
                    (self.rect.right - 15, self.rect.bottom - 10),
                    (self.rect.right - 10, self.rect.bottom - 15),
                    (self.rect.right - 20, self.rect.bottom - 15)
                ])
    
    def update(self, mouse_pos):
        """Update browser state based on mouse position"""
        # Check if mouse is over file list area
        list_rect = pygame.Rect(
            self.rect.x + 5,
            self.rect.y + 45,
            self.rect.width - 10,
            self.rect.height - 55
        )
        
        if list_rect.collidepoint(mouse_pos):
            # Calculate which file the mouse is hovering over
            relative_y = mouse_pos[1] - list_rect.y + self.scroll_y
            hover_index = int(relative_y // self.item_height)
            
            # Check if hover index is valid
            if 0 <= hover_index < len(self.files):
                self.hover_index = hover_index
            else:
                self.hover_index = -1
        else:
            self.hover_index = -1
    
    def handle_event(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is inside file list area
            list_rect = pygame.Rect(
                self.rect.x + 5,
                self.rect.y + 45,
                self.rect.width - 10,
                self.rect.height - 55
            )
            
            if list_rect.collidepoint(event.pos):
                # Handle scrolling
                if event.button == 4:  # Scroll up
                    self.scroll_y = max(0, self.scroll_y - self.scroll_speed)
                    return True
                
                elif event.button == 5:  # Scroll down
                    self.scroll_y = min(self.max_scroll, self.scroll_y + self.scroll_speed)
                    return True
                
                # Handle file selection
                elif event.button == 1:  # Left click
                    relative_y = event.pos[1] - list_rect.y + self.scroll_y
                    clicked_index = int(relative_y // self.item_height)
                    
                    # Check if clicked index is valid
                    if 0 <= clicked_index < len(self.files):
                        # Update selection
                        self.selected_index = clicked_index
                        self.selected_file = self.files[clicked_index][1]
                        
                        # Notify callback if set
                        if self.file_selected_callback:
                            self.file_selected_callback(self.selected_file)
                        
                        return True
            
            # Check if click is on refresh button (not implemented yet)
            
        # Handle mouse wheel for scrolling
        elif event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_y = max(0, min(self.max_scroll, self.scroll_y - event.y * self.scroll_speed))
                return True
        
        return False

def add_midi_file_browser(screen_width, keyboard_rect_height, ui_panels_height):
    """Create a file browser panel for MIDI files"""
    browser_width = 300
    browser_height = 400
    
    # Position on the right side of the screen
    browser_x = screen_width - browser_width - 20
    browser_y = keyboard_rect_height + 20
    
    # Create a file browser
    file_browser = FileBrowser(
        browser_x, browser_y, browser_width, browser_height,
        directory=RECORDINGS_FOLDER, file_extension=".mid"
    )
    
    # Create a UI panel to hold the browser
    browser_panel = UIPanel(
        browser_x - 10, browser_y - 10, 
        browser_width + 20, browser_height + 20,
        bg_color=(35, 37, 45)
    )
    browser_panel.set_title("MIDI File Browser")
    
    # Add refresh and play buttons for the file browser
    button_y = browser_y + browser_height + 15
    button_width = 140
    button_height = 40
    
    refresh_button = Button(
        browser_x, button_y,
        button_width, button_height,
        (80, 140, 200, 255),
        "REFRESH",
        text_color=WHITE,
        icon="retry"
    )
    
    load_button = Button(
        browser_x + browser_width - button_width, button_y,
        button_width, button_height,
        (100, 180, 100, 255),
        "LOAD",
        text_color=WHITE,
        icon="play"
    )
    
    return file_browser, browser_panel, refresh_button, load_button

    # Create slider for temperature control
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, 
                 integer_only=False, primary_color=(100, 140, 230, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_radius = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.integer_only = integer_only
        self.dragging = False
        self.is_hovered = False
        
        # Modern design properties
        self.primary_color = primary_color
        self.track_color = (60, 62, 68)
        self.track_active_color = primary_color
        self.handle_color = (230, 230, 240)
        self.handle_hover_color = (255, 255, 255)
        self.text_color = (220, 220, 230)
        self.value_color = primary_color[:3] + (255,)  # Alpha always 255 for text
        
        # Animation properties
        self.current_handle_color = self.handle_color
        self.color_transition_speed = 0.15
        self.pulse_alpha = 0
        self.pulse_size = 0
        
        # Font setup
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.label_font = pygame.font.SysFont(self.font_name, 16)
        self.value_font = pygame.font.SysFont(self.font_name, 14, bold=True)
        self.bounds_font = pygame.font.SysFont(self.font_name, 12)
        
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
    
    def _interpolate_color(self, color1, color2, fraction):
        """Smoothly interpolate between two colors"""
        r1, g1, b1 = color1[:3]
        r2, g2, b2 = color2[:3]
        
        r = int(r1 + (r2 - r1) * fraction)
        g = int(g1 + (g2 - g1) * fraction)
        b = int(b1 + (b2 - b1) * fraction)
        
        alpha = color1[3] if len(color1) > 3 else 255
        return (r, g, b, alpha)
        
    def update(self, mouse_pos, dt=1/60):
        """Update slider state based on mouse position"""
        # Check if mouse is hovering over handle
        mouse_x, mouse_y = mouse_pos
        handle_rect = pygame.Rect(self.handle_x - self.handle_radius, 
                                 self.rect.centery - self.handle_radius,
                                 self.handle_radius * 2, 
                                 self.handle_radius * 2)
        
        prev_hovering = self.is_hovered
        self.is_hovered = handle_rect.collidepoint(mouse_x, mouse_y)
        
        # Visual pulse effect when hover starts
        if not prev_hovering and self.is_hovered:
            self.pulse_alpha = 120
            self.pulse_size = 0
        
        # Animate pulse effect
        if self.pulse_alpha > 0:
            self.pulse_alpha = max(0, self.pulse_alpha - 240 * dt)
            self.pulse_size = min(12, self.pulse_size + 30 * dt)
        
        # Animate handle color
        target_color = self.handle_hover_color if self.is_hovered or self.dragging else self.handle_color
        self.current_handle_color = self._interpolate_color(
            self.current_handle_color, target_color, self.color_transition_speed)
        
    def draw(self, screen):
        # Draw background track
        track_height = max(4, self.rect.height)  # Minimum height of 4 pixels
        track_rect = pygame.Rect(
            self.rect.x, 
            self.rect.centery - track_height // 2,
            self.rect.width, 
            track_height
        )
        pygame.draw.rect(screen, self.track_color, track_rect, border_radius=track_height//2)
        
        # Draw active part of the track
        active_width = self.handle_x - self.rect.x
        if active_width > 0:
            active_rect = pygame.Rect(
                self.rect.x, 
                self.rect.centery - track_height // 2,
                active_width, 
                track_height
            )
            pygame.draw.rect(screen, self.track_active_color, active_rect, border_radius=track_height//2)
        
        # Draw label above slider
        label_text = self.label_font.render(f"{self.label}", True, self.text_color)
        
        # Format value based on integer-only mode
        if self.integer_only:
            value_text = self.value_font.render(f"{self.value}", True, self.value_color)
        else:
            value_text = self.value_font.render(f"{self.value:.2f}", True, self.value_color)
        
        # Position label centered above slider
        label_x = self.rect.x + (self.rect.width - label_text.get_width()) // 2
        screen.blit(label_text, (label_x, self.rect.y - 40))

        # Position value text below the label
        value_x = self.rect.x + (self.rect.width - value_text.get_width()) // 2
        screen.blit(value_text, (value_x, self.rect.y - 20))
        
        # Draw min/max values as smaller text
        # Format based on integer-only mode
        if self.integer_only:
            min_text = self.bounds_font.render(f"{int(self.min_val)}", True, (150, 150, 160))
            max_text = self.bounds_font.render(f"{int(self.max_val)}", True, (150, 150, 160))
        else:
            min_text = self.bounds_font.render(f"{self.min_val:.1f}", True, (150, 150, 160))
            max_text = self.bounds_font.render(f"{self.max_val:.1f}", True, (150, 150, 160))
            
        screen.blit(min_text, (self.rect.x - 5, self.rect.y + self.rect.height + 5))
        screen.blit(max_text, (self.rect.x + self.rect.width - max_text.get_width() + 5, 
                             self.rect.y + self.rect.height + 5))
        
        # Draw pulse effect when hovering begins
        if self.pulse_alpha > 0:
            pulse_radius = self.handle_radius + self.pulse_size
            pulse_color = self.primary_color[:3] + (self.pulse_alpha,)
            pygame.gfxdraw.filled_circle(screen, self.handle_x, self.rect.centery, 
                                       int(pulse_radius), pulse_color)
        
        # Draw handle shadow
        shadow_offset = 2
        shadow_color = (30, 30, 35, 100)
        pygame.gfxdraw.filled_circle(screen, self.handle_x, self.rect.centery + shadow_offset, 
                                   self.handle_radius - 1, shadow_color)
        
        # Draw handle with animated color
        pygame.gfxdraw.filled_circle(screen, self.handle_x, self.rect.centery, 
                                   self.handle_radius - 1, self.current_handle_color)
        
        # Add subtle highlight to top of handle for 3D effect
        if self.is_hovered or self.dragging:
            highlight_color = (255, 255, 255, 80)
            highlight_radius = self.handle_radius - 4
            highlight_offset = -2
            pygame.draw.circle(screen, highlight_color, 
                             (self.handle_x, self.rect.centery + highlight_offset), 
                             highlight_radius, width=1)
        
    def handle_event(self, event):
        """Handle mouse events for the slider"""
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
    screen_height = keyboard.rect.height + control_panel_height + ai_panel_height + 250  # Add extra space (250px) for piano roll

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
    
    # Piano roll image placeholder
    piano_roll_image = None
    
    # Function to update piano roll visualization
    def update_piano_roll_image(midi_path):
        nonlocal piano_roll_image
        if not os.path.exists(midi_path):
            return False
            
        try:
            # Load MIDI file into PrettyMIDI object
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Create a temporary file path for the image
            image_path = os.path.join(recorder.recordings_dir, "temp_piano_roll.png")
            
            # Retrieve piano roll and save as image
            piano_roll = midi_data.get_piano_roll()
            plt.figure(figsize=(10, 3))
            plt.imshow(piano_roll, aspect='auto', cmap='Blues_r', origin='lower')
            plt.title("MIDI Piano Roll")
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            
            # Load the image into pygame and scale it
            piano_roll_image = pygame.image.load(image_path)
            piano_roll_image = pygame.transform.scale(
                piano_roll_image, 
                (screen.get_width() - 40, 200)
            )
            
            if status:
                status.set_message("Piano roll updated", (100, 200, 255))
                
            return True
            
        except Exception as e:
            print(f"Error generating piano roll: {e}")
            if status:
                status.set_message(f"Error creating piano roll: {str(e)}", (255, 100, 100))
            return False
    
    # Create recording indicator animation variables
    recording_indicator_size = 10
    recording_indicator_alpha = 255
    recording_indicator_growing = False
    
    # Track AI generation state
    ai_generation_active = False
    
    # Frame timing for smoother animations
    clock = pygame.time.Clock()
    
    while playing:
        buttons[0].is_active = recorder.recording
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
                        
                    # In the Save MIDI button click handler (around line 2930):
                    elif buttons[2].is_clicked(pos):  # Save MIDI button
                        if recorder.save_midi_recording(anchor_note):
                            if status:
                                status.set_message(f"MIDI recording saved to {RECORDINGS_FOLDER}", (100, 100, 255))
                            # Add this line to update piano roll
                            update_piano_roll_image(recorder.midi_file_path)
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
                                    update_piano_roll_image(combined_path)
                                    
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
        
        # Animation for record button
        if recorder.recording and buttons[0].icon == "record":
            # Change the button color to pulse
            pulse_amt = (math.sin(pygame.time.get_ticks() / 200) + 1) / 2  # 0 to 1
            buttons[0].color = (
                255,  # Full red
                int(40 * pulse_amt),  # Pulsing green
                int(40 * pulse_amt),  # Pulsing blue
                255
            )
        else:
            # Reset to original color when not recording
            buttons[0].color = (220, 60, 60, 255)  # Original red

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
            # Position it above the piano roll
            status_y = ai_panel.rect.y + ai_panel.rect.height - 40  # At the bottom of AI panel
            status.draw(screen, (status_x, status_y))
        
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
        
        if recorder.recording:
            # Position next to the Record button
            rec_x = buttons[0].rect.right + 10
            rec_y = buttons[0].rect.centery - 8
            
            # Pulsing effect
            alpha = 128 + int(127 * math.sin(time.time() * 4))
            
            # Draw "REC" text
            rec_font = pygame.font.SysFont("Arial", 16, bold=True)
            rec_text = rec_font.render("● REC", True, (255, 50, 50))
            rec_text.set_alpha(alpha)
            screen.blit(rec_text, (rec_x, rec_y))
            
        if piano_roll_image:
            # Position it in the newly created space at the bottom
            piano_roll_rect = piano_roll_image.get_rect()
            piano_roll_rect.centerx = screen.get_rect().centerx
            
            # Place it below the AI panel
            ai_panel_bottom = ai_panel.rect.y + ai_panel.rect.height
            piano_roll_rect.top = ai_panel_bottom + 20  # 20px margin
            
            # Draw a background panel for the piano roll
            bg_rect = piano_roll_rect.inflate(20, 20)
            pygame.draw.rect(screen, (25, 27, 35), bg_rect, border_radius=10)
            pygame.draw.rect(screen, (60, 65, 70), bg_rect, width=1, border_radius=10)
            
            # Draw a title for the piano roll
            roll_title_font = pygame.font.SysFont("Arial", 16, bold=True)
            roll_title = roll_title_font.render("MIDI Piano Roll", True, (220, 220, 230))
            # roll_title_rect = roll_title.get_rect(centerx=piano_roll_rect.centerx, bottom=piano_roll_rect.top - 8)
            # screen.blit(roll_title, roll_title_rect)
            
            # Draw the image
            screen.blit(piano_roll_image, piano_roll_rect)
                    
        pygame.display.update()
        
        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
    print("Goodbye")
class NoteRecorder:
    def __init__(self, framerate_hz, channels, tones):
        self.midi_file_path = None  # Add this to track the current MIDI file
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
    
    def save_midi_recording(self, anchor_note=None, output_path=None):
        if not self.midi_notes:
            print("No MIDI notes to save.")
            return False
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pianoputer_midi_{timestamp}.mid"
            output_path = os.path.join(self.recordings_dir, filename)
        
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
        mid.save(output_path)
        self.midi_file_path = output_path
        print(f"MIDI recording saved to: {output_path}")
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