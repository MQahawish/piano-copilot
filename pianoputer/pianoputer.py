#!/usr/bin/env python
"""
Refactored Pianoputer AI application.
"""

# ====================================================
# Standard Library Imports
# ====================================================
import argparse
import codecs
import datetime
import math
import os
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import torch
# ====================================================
# Third-Party Imports
# ====================================================
import matplotlib.pyplot as plt
import numpy
import pretty_midi
import librosa
import pygame
import pygame.gfxdraw
import soundfile
import keyboardlayout as kl
import keyboardlayout.pygame as klp

# ====================================================
# Local Imports
# ====================================================
from components.button import Button
from components.file_browser import FileBrowser
from components.status_message import StatusMessage
from components.ui_panel import UIPanel
from components.slider import Slider
from components.midi_player import MIDIPlayer
from components.note_recorder import NoteRecorder
from components.config import *
from ai_composer import AIComposer

# ====================================================
# Argument Parsing Functions
# ====================================================
def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
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


def process_args(parser: argparse.ArgumentParser, args: Optional[List] = None) -> Tuple[str, str, bool]:
    """Process command-line arguments."""
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    if not args.verbose:
        warnings.simplefilter("ignore")

    wav_path = args.wav
    if wav_path.startswith(AUDIO_ASSET_PREFIX):
        wav_path = os.path.join(CURRENT_WORKING_DIR, wav_path)

    keyboard_path = args.keyboard
    if keyboard_path.startswith(KEYBOARD_ASSET_PREFIX):
        keyboard_path = os.path.join(CURRENT_WORKING_DIR, keyboard_path)
    return wav_path, keyboard_path, args.clear_cache


# ====================================================
# Audio Utility Functions
# ====================================================
def get_audio_data(wav_path: str) -> Tuple:
    """Read the WAV file and return audio data, sample rate, and channel count."""
    audio_data, framerate_hz = soundfile.read(wav_path)
    array_shape = audio_data.shape
    channels = 1 if len(array_shape) == 1 else array_shape[1]
    return audio_data, framerate_hz, channels


def get_or_create_key_sounds(
    wav_path: str,
    sample_rate_hz: int,
    channels: int,
    tones: List[int],
    clear_cache: bool,
    keys: List[str],
) -> Generator[pygame.mixer.Sound, None, None]:
    """
    Generate or load cached sound samples for each key.
    Returns a generator of pygame sound objects.
    """
    sounds = []
    y, sr = librosa.load(wav_path, sr=sample_rate_hz, mono=(channels == 1))
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
        if cached_path.exists():
            print("Loading note {} out of {} for {}".format(i + 1, len(tones), keys[i]))
            sound, sr = librosa.load(str(cached_path), sr=sample_rate_hz, mono=(channels == 1))
            if channels > 1:
                sound = numpy.transpose(sound)
        else:
            print("Transposing note {} out of {} for {}".format(i + 1, len(tones), keys[i]))
            if channels == 1:
                sound = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=tone)
            else:
                new_channels = [
                    librosa.effects.pitch_shift(y=y[i], sr=sr, n_steps=tone)
                    for i in range(channels)
                ]
                sound = numpy.ascontiguousarray(numpy.vstack(new_channels).T)
            soundfile.write(str(cached_path), sound, sample_rate_hz, DESCRIPTOR_32BIT)
        sounds.append(sound)
    return map(pygame.sndarray.make_sound, sounds)


# ====================================================
# Keyboard Configuration Functions
# ====================================================
BLACK_INDICES_C_SCALE = [1, 3, 6, 8, 10]
LETTER_KEYS_TO_INDEX = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def __get_black_key_indices(key_name: str) -> set:
    """Return a set of indices that represent black keys based on the anchor note."""
    letter_key_index = LETTER_KEYS_TO_INDEX[key_name]
    black_key_indices = set()
    for ind in BLACK_INDICES_C_SCALE:
        new_index = ind - letter_key_index
        if new_index < 0:
            new_index += 12
        black_key_indices.add(new_index)
    return black_key_indices


def get_keyboard_info(keyboard_file: str):
    """
    Parse the keyboard file and return keys, tones, color mapping,
    key colors, text colors, and the anchor note.
    """
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
        # In anchor mode, keys go up in half steps and are colored from grey low to white high.
        rgb_val = 255 - (len(keys) - 1 - index) * 3
        color = (rgb_val, rgb_val, rgb_val, 255)
        color_to_key[color].append(key)

    return keys, tones, color_to_key, key_color, key_txt_color, anchor_note


# ====================================================
# UI Helper Functions
# ====================================================
def add_midi_file_browser(screen_width, keyboard_rect_height, ui_panels_height):
    """Create and return the MIDI file browser UI elements."""
    browser_width = 300
    browser_height = 400
    browser_x = screen_width - browser_width - 20
    browser_y = keyboard_rect_height + 20

    file_browser = FileBrowser(
        browser_x, browser_y, browser_width, browser_height,
        directory=RECORDINGS_FOLDER, file_extension=".mid"
    )

    browser_panel = UIPanel(
        browser_x - 10, browser_y - 10,
        browser_width + 20, browser_height + 20,
        bg_color=(35, 37, 45)
    )
    browser_panel.set_title("MIDI File Browser")

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


def configure_pygame_audio_and_set_ui(
    framerate_hz: int,
    channels: int,
    keyboard_arg: str,
    color_to_key: Dict[str, List[kl.Key]],
    key_color: Tuple[int, int, int, int],
    key_txt_color: Tuple[int, int, int, int],
) -> Tuple[pygame.Surface, klp.KeyboardLayout, List[Button], UIPanel, UIPanel, StatusMessage, Slider, Slider]:
    """Initialize pygame, audio, UI panels, keyboard layout, and control buttons/sliders."""
    pygame.display.init()
    pygame.display.set_caption("Pianoputer AI")
    pygame.event.set_blocked(None)
    pygame.event.set_allowed(list(ALLOWED_EVENTS))
    pygame.font.init()
    pygame.mixer.init(
        framerate_hz,
        BITS_32BIT,
        channels,
        allowedchanges=AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED,
    )

    # Determine keyboard layout based on filename
    if "qwerty" in keyboard_arg:
        layout_name = kl.LayoutName.QWERTY
    elif "azerty" in keyboard_arg:
        layout_name = kl.LayoutName.AZERTY_LAPTOP
    else:
        raise ValueError("keyboard must have qwerty or azerty in its name")

    margin = 4
    key_size = 60
    overrides = {}
    for color_value, keys in color_to_key.items():
        override_color = pygame.Color(color_value)
        inverted_color = list(~override_color)
        other_val = 255
        if (abs(color_value[0] - inverted_color[0]) > abs(color_value[0] - other_val)) or color_value == CYAN:
            override_txt_color = pygame.Color(inverted_color)
        else:
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
    letter_key_size = (key_size, key_size)
    keyboard = klp.KeyboardLayout(
        layout_name, keyboard_info, letter_key_size, key_info, overrides
    )

    control_panel_height = 120
    ai_panel_height = 300
    screen_width = max(keyboard.rect.width, 800)
    screen_height = keyboard.rect.height + control_panel_height + ai_panel_height + 250
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((20, 20, 25))

    # Recording controls panel
    panel = UIPanel(10, keyboard.rect.height + 10, screen_width - 20, control_panel_height - 20, bg_color=(40, 40, 45))
    panel.set_title("Recording Controls")

    button_width = 120
    button_height = 40
    button_spacing = (screen_width - 60 - (button_width * 3)) // 2
    start_x = 30
    button_y_offset = keyboard.rect.height + 45

    record_button = Button(
        start_x, 
        button_y_offset, 
        button_width, 
        button_height, 
        (220, 60, 60, 255),
        "RECORD",
        icon="record"
    )
    stop_button = Button(
        start_x + button_width + button_spacing, 
        button_y_offset, 
        button_width, 
        button_height, 
        (70, 70, 80, 255),
        "STOP",
        icon="stop",
        text_color=WHITE
    )
    save_midi_button = Button(
        start_x + (button_width + button_spacing) * 2, 
        button_y_offset, 
        button_width, 
        button_height, 
        (70, 105, 200, 255),
        "SAVE MIDI",
        icon="save_midi"
    )
    panel.add_element(record_button)
    panel.add_element(stop_button)
    panel.add_element(save_midi_button)

    # AI controls panel
    ai_panel_y = keyboard.rect.height + control_panel_height + 10
    ai_panel = UIPanel(10, ai_panel_y, screen_width - 20, ai_panel_height - 20, bg_color=(35, 40, 50))
    ai_panel.set_title("AI Composition")

    ai_button_y = ai_panel_y + 45
    generate_button = Button(
        start_x, 
        ai_button_y,
        button_width, 
        button_height, 
        (100, 130, 220, 255),
        "GENERATE",
        text_color=WHITE
    )
    accept_button = Button(
        start_x + button_width + button_spacing, 
        ai_button_y,
        button_width, 
        button_height, 
        (100, 220, 100, 255),
        "ACCEPT",
        text_color=WHITE
    )
    retry_button = Button(
        start_x + (button_width + button_spacing) * 2, 
        ai_button_y,
        button_width, 
        button_height, 
        (220, 160, 40, 255),
        "RETRY",
        text_color=WHITE
    )
    playback_y = ai_button_y + button_height + 20
    play_button = Button(
        start_x, 
        playback_y,
        button_width // 2, 
        button_height, 
        (100, 180, 100, 255),
        "PLAY",
        text_color=WHITE
    )
    pause_button = Button(
        start_x + button_width // 2, 
        playback_y,
        button_width // 2, 
        button_height, 
        (70, 70, 80, 255),
        "PAUSE",
        text_color=WHITE
    )

    slider_width = screen_width - 140
    slider_x = (screen_width - slider_width) // 2
    temp_slider_y = playback_y + button_height + 30
    tokens_slider_y = temp_slider_y + 60
    temp_slider = Slider(
        slider_x, 
        temp_slider_y,
        slider_width, 
        10, 
        0.5, 1.5, 0.9, 
        "Temperature",
        integer_only=False
    )
    tokens_slider = Slider(
        slider_x, 
        tokens_slider_y,
        slider_width, 
        10, 
        64, 512, 256, 
        "Max Tokens",
        integer_only=True
    )

    ai_panel.add_element(generate_button)
    ai_panel.add_element(accept_button)
    ai_panel.add_element(retry_button)
    ai_panel.add_element(play_button)
    ai_panel.add_element(pause_button)

    buttons = [
        record_button, stop_button, save_midi_button,
        generate_button, accept_button, retry_button,
        play_button, pause_button
    ]

    status = StatusMessage()
    keyboard.draw(screen)
    panel.draw(screen)
    ai_panel.draw(screen)
    temp_slider.draw(screen)
    tokens_slider.draw(screen)
    pygame.display.update()

    return screen, keyboard, buttons, panel, ai_panel, status, temp_slider, tokens_slider


def update_piano_roll_image(midi_path: str, recorder, screen: pygame.Surface, status: Optional[StatusMessage]) -> Optional[pygame.Surface]:
    """
    Generate and return a new piano roll image from a MIDI file.
    """
    if not os.path.exists(midi_path):
        return None

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        image_path = os.path.join(recorder.recordings_dir, "temp_piano_roll.png")
        piano_roll = midi_data.get_piano_roll()
        plt.figure(figsize=(10, 3))
        plt.imshow(piano_roll, aspect='auto', cmap='Blues_r', origin='lower')
        plt.title("MIDI Piano Roll")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        piano_roll_image = pygame.image.load(image_path)
        piano_roll_image = pygame.transform.scale(piano_roll_image, (screen.get_width() - 40, 200))
        if status:
            status.set_message("Piano roll updated", (100, 200, 255))
        return piano_roll_image

    except Exception as e:
        print(f"Error generating piano roll: {e}")
        if status:
            status.set_message(f"Error creating piano roll: {str(e)}", (255, 100, 100))
        return None


# ====================================================
# Main Event Loop and Application Logic
# ====================================================
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
    ai_composer = None,
):
    """
    Main loop handling user events including key presses, button clicks,
    recording, and AI composition generation.
    """
    sound_by_key = dict(zip(keys, key_sounds))
    playing = True
    recorder = NoteRecorder(framerate_hz, channels, tones)
    midi_player = MIDIPlayer(status_callback=status.set_message if status else None)
    piano_roll_image = None

    if ai_composer:
        recorder.ai_composer = ai_composer

    clock = pygame.time.Clock()

    ai_generation_active = False

    while playing:
        buttons[0].is_active = recorder.recording
        mouse_pos = pygame.mouse.get_pos()

        for button in buttons:
            button.update(mouse_pos)
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
                recorder.note_up(key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                slider_handled = False
                if temp_slider:
                    slider_handled = temp_slider.handle_event(event)
                if not slider_handled and tokens_slider:
                    slider_handled = tokens_slider.handle_event(event)

                if not slider_handled:
                    if buttons[0].is_clicked(pos):  # Record button
                        recorder.start_recording()
                        if status:
                            status.set_message("Recording started...", (255, 100, 100))
                        buttons[0].is_active = True
                        buttons[1].is_active = False

                    elif buttons[1].is_clicked(pos):  # Stop button
                        recorder.stop_recording()
                        if status:
                            status.set_message("Recording stopped. Ready to save or generate.", (100, 255, 100))
                        buttons[0].is_active = False
                        buttons[1].is_active = True

                    elif buttons[2].is_clicked(pos):  # Save MIDI button
                        if recorder.save_midi_recording(anchor_note):
                            if status:
                                status.set_message(f"MIDI recording saved to {RECORDINGS_FOLDER}", (100, 100, 255))
                            piano_roll_image = update_piano_roll_image(recorder.midi_file_path, recorder, screen, status)
                        else:
                            if status:
                                status.set_message("No recording to save", (255, 100, 100))

                    # AI Generate
                    elif buttons[3].is_clicked(pos):
                        if not recorder.midi_notes:
                            if status:
                                status.set_message("Record something first!", (255, 100, 100))
                        else:
                            if status:
                                status.set_message("Generating AI continuation...", (100, 100, 255))
                            buttons[3].is_active = True
                            ai_generation_active = True
                            temperature = temp_slider.value if temp_slider else 0.9
                            max_tokens = int(tokens_slider.value) if tokens_slider else 256

                            try:
                                import threading

                                def generate_task():
                                    temp_midi_path = os.path.join(recorder.recordings_dir, "temp_recording.mid")
                                    recorder.save_midi_recording(output_path=temp_midi_path, anchor_note=anchor_note)
                                    encoded_input = recorder.ai_composer.process_user_recording(temp_midi_path)
                                    generated_output = recorder.ai_composer.generate_continuation(
                                        encoded_input, temperature=temperature, top_k=100, max_new_tokens=max_tokens
                                    )
                                    recorder.current_continuation = generated_output
                                    combined_path = os.path.join(recorder.recordings_dir, "ai_continuation.mid")
                                    recorder.ai_composer.create_combined_midi(
                                        encoded_input, generated_output, combined_path
                                    )
                                    nonlocal piano_roll_image
                                    piano_roll_image = update_piano_roll_image(combined_path, recorder, screen, status)
                                    nonlocal ai_generation_active
                                    ai_generation_active = False
                                    buttons[3].is_active = False
                                    if status:
                                        status.set_message("AI continuation ready! Press PLAY to listen.", (100, 255, 100))
                                    midi_player.load_midi(combined_path)

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
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                final_path = os.path.join(recorder.recordings_dir, f"composition_{timestamp}.mid")
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
                            if status:
                                status.set_message("Generating new AI continuation...", (220, 160, 40))
                            buttons[5].is_active = True
                            ai_generation_active = True
                            temperature = min(1.2, temp_slider.value + 0.1) if temp_slider else 1.0
                            max_tokens = int(tokens_slider.value) if tokens_slider else 256

                            try:
                                import threading

                                def retry_task():
                                    temp_midi_path = os.path.join(recorder.recordings_dir, "temp_recording.mid")
                                    recorder.save_midi_recording(output_path=temp_midi_path, anchor_note=anchor_note)
                                    encoded_input = recorder.ai_composer.process_user_recording(temp_midi_path)
                                    generated_output = recorder.ai_composer.generate_continuation(
                                        encoded_input, temperature=temperature, top_k=150, max_new_tokens=max_tokens
                                    )
                                    recorder.current_continuation = generated_output
                                    combined_path = os.path.join(recorder.recordings_dir, "ai_continuation.mid")
                                    recorder.ai_composer.create_combined_midi(
                                        encoded_input, generated_output, combined_path
                                    )
                                    nonlocal piano_roll_image
                                    piano_roll_image = update_piano_roll_image(combined_path, recorder, screen, status)
                                    nonlocal ai_generation_active
                                    ai_generation_active = False
                                    buttons[5].is_active = False
                                    if status:
                                        status.set_message("New AI continuation ready! Press PLAY to listen.", (220, 160, 40))
                                    midi_player.load_midi(combined_path)

                                threading.Thread(target=retry_task).start()

                            except Exception as e:
                                ai_generation_active = False
                                buttons[5].is_active = False
                                if status:
                                    status.set_message(f"Error generating: {str(e)}", (255, 100, 100))

                    elif buttons[6].is_clicked(pos):  # Play button
                        midi_player.play()

                    elif buttons[7].is_clicked(pos):  # Pause button
                        midi_player.pause()

            elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
                if temp_slider:
                    temp_slider.handle_event(event)
                if tokens_slider:
                    tokens_slider.handle_event(event)

        # Animation for record button
        if recorder.recording and buttons[0].icon == "record":
            pulse_amt = (math.sin(pygame.time.get_ticks() / 200) + 1) / 2
            buttons[0].color = (255, int(40 * pulse_amt), int(40 * pulse_amt), 255)
        else:
            buttons[0].color = (220, 60, 60, 255)

        screen.fill((20, 20, 25))
        keyboard.draw(screen)
        if panel:
            panel.draw(screen)
        if ai_panel:
            ai_panel.draw(screen)
        if temp_slider:
            temp_slider.draw(screen)
        if tokens_slider:
            tokens_slider.draw(screen)
        if status:
            status_x = (screen.get_width() - 300) // 2
            status_y = ai_panel.rect.y + ai_panel.rect.height - 40
            status.draw(screen, (status_x, status_y))

        if ai_generation_active:
            spinner_x = screen.get_width() // 2
            spinner_y = ai_panel.rect.y + (ai_panel.rect.height // 2)
            current_time = pygame.time.get_ticks()
            angle = (current_time % 1000) / 1000 * 360
            spinner_radius = 20
            for i in range(8):
                dot_angle = angle + i * 45
                dot_x = spinner_x + int(math.cos(math.radians(dot_angle)) * spinner_radius)
                dot_y = spinner_y + int(math.sin(math.radians(dot_angle)) * spinner_radius)
                alpha = 255 - ((i * 25) % 255)
                pygame.gfxdraw.filled_circle(screen, dot_x, dot_y, 4, (100, 180, 255, alpha))
            gen_font = pygame.font.SysFont("Arial", 16, bold=True)
            gen_text = gen_font.render("Generating AI Continuation...", True, (200, 200, 255))
            screen.blit(gen_text, (spinner_x - gen_text.get_width() // 2, spinner_y + 30))

        if recorder.recording:
            rec_x = buttons[0].rect.right + 10
            rec_y = buttons[0].rect.centery - 8
            alpha = 128 + int(127 * math.sin(time.time() * 4))
            rec_font = pygame.font.SysFont("Arial", 16, bold=True)
            rec_text = rec_font.render("● REC", True, (255, 50, 50))
            rec_text.set_alpha(alpha)
            screen.blit(rec_text, (rec_x, rec_y))

        if piano_roll_image:
            piano_roll_rect = piano_roll_image.get_rect()
            piano_roll_rect.centerx = screen.get_rect().centerx
            ai_panel_bottom = ai_panel.rect.y + ai_panel.rect.height
            piano_roll_rect.top = ai_panel_bottom + 20
            bg_rect = piano_roll_rect.inflate(20, 20)
            pygame.draw.rect(screen, (25, 27, 35), bg_rect, border_radius=10)
            pygame.draw.rect(screen, (60, 65, 70), bg_rect, width=1, border_radius=10)
            screen.blit(piano_roll_image, piano_roll_rect)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    print("Goodbye")


# ====================================================
# Main Application Function
# ====================================================
def play_pianoputer(args: Optional[List[str]] = None):
    parser = get_parser()
    wav_path, keyboard_path, clear_cache = process_args(parser, args)
    audio_data, framerate_hz, channels = get_audio_data(wav_path)
    keys, tones, color_to_key, key_color, key_txt_color, anchor_note = get_keyboard_info(keyboard_path)
    key_sounds = get_or_create_key_sounds(wav_path, framerate_hz, channels, tones, clear_cache, keys)
    screen, keyboard, buttons, panel, ai_panel, status, temp_slider, tokens_slider = configure_pygame_audio_and_set_ui(
        framerate_hz, channels, keyboard_path, color_to_key, key_color, key_txt_color
    )
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_checkpoint_dir = os.path.join(BASE_DIR, "model", "ckpts", "3layers12heads32batch512seq-simple")
    tokenizer_path = os.path.join(BASE_DIR, "model", "simple_tokenizer.json")
    ai_composer = AIComposer(model_checkpoint_dir, tokenizer_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Ready for you to play! ...")
    
    play_until_user_exits(
        keys, list(key_sounds), keyboard, screen, buttons,
        framerate_hz, channels, tones, anchor_note,
        panel, ai_panel, status, temp_slider, tokens_slider, ai_composer
    )


if __name__ == "__main__":
    play_pianoputer()
