
import os
import sys
import datetime
import pygame
from pathlib import Path
from .config import CURRENT_WORKING_DIR, RECORDINGS_FOLDER
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
