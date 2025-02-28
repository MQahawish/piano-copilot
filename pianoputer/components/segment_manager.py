import pygame

class SegmentManagerUI:
    """UI manager for handling MIDI segments with visual representation"""
    
    def __init__(self, x, y, width, height, recordings_dir, 
                 status_callback=None, on_segment_selected=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.recordings_dir = recordings_dir
        self.status_callback = status_callback
        self.on_segment_selected = on_segment_selected
        
        # UI properties
        self.bg_color = (30, 32, 40)
        self.border_color = (60, 62, 70, 180)
        self.title = "Composition Timeline"
        self.border_radius = 10
        
        # Font setup
        self.font_name = "Inter" if "Inter" in pygame.font.get_fonts() else "Arial"
        self.title_font = pygame.font.SysFont(self.font_name, 18, bold=True)
        self.label_font = pygame.font.SysFont(self.font_name, 14)
        self.segment_font = pygame.font.SysFont(self.font_name, 12)
        
        # Segment display properties
        self.segment_height = 40
        self.segment_spacing = 10
        self.segment_colors = {
            "recorded": (100, 180, 255),  # Blue
            "generated": (180, 130, 255),  # Purple
            "accepted": (100, 220, 130),  # Green
            "rejected": (200, 100, 100),  # Red
            "pending": (150, 150, 150)    # Gray
        }
        
        # Interaction properties
        self.selected_segment_index = -1
        self.hover_segment_index = -1
        self.is_dragging = False
        self.drag_start_x = 0
        
        # Timeline properties (for scrolling large compositions)
        self.timeline_offset = 0
        self.visible_width = width - 20  # Adjust for padding
        self.total_width = 0
        self.scale_factor = 100  # Pixels per second of music
        
        # Animation properties
        self.hover_alpha = 0
        self.pulse_progress = 0
        
        # Segments data (will be populated from AIComposer)
        self.segments = []
        
    def update_segments(self, ai_composer):
        """Update the segment display data from the AI composer"""
        self.segments = ai_composer.get_all_segments()
        self.selected_segment_index = ai_composer.current_segment_index
        
        # Calculate total timeline width
        if self.segments:
            self.total_width = sum(self._get_segment_width(segment) for segment in self.segments) + \
                              ((len(self.segments) - 1) * self.segment_spacing)
        else:
            self.total_width = 0
    
    def _get_segment_width(self, segment):
        """Calculate the width of a segment based on its duration"""
        # In a real implementation, you'd extract the duration from the MIDI
        # For now, use a placeholder width
        base_width = 120
        
        # If we have a continuation, make it wider
        if segment.get("continuation"):
            base_width += 80
            
        return base_width
    
    def draw(self, screen):
        """Draw the timeline with all segments"""
        # Draw panel background with rounded corners
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(screen, self.border_color, self.rect, width=1, border_radius=self.border_radius)
        
        # Draw title
        title_text = self.title_font.render(self.title, True, (220, 220, 230))
        title_rect = title_text.get_rect(
            midtop=(self.rect.centerx, self.rect.y + 10)
        )
        screen.blit(title_text, title_rect)
        
        # Draw separator line
        pygame.draw.line(screen, self.border_color, 
                     (self.rect.x + 10, self.rect.y + 35), 
                     (self.rect.right - 10, self.rect.y + 35),
                     width=1)
        
        # If no segments, show message
        if not self.segments:
            info_text = self.label_font.render("No segments recorded yet", True, (180, 180, 180))
            info_rect = info_text.get_rect(
                center=(self.rect.centerx, self.rect.y + 70)
            )
            screen.blit(info_text, info_rect)
            return
        
        # Set up clipping to prevent drawing outside the panel
        content_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.y + 45,
            self.rect.width - 20,
            self.rect.height - 55
        )
        
        original_clip = screen.get_clip()
        screen.set_clip(content_rect)
        
        # Draw segments
        x_pos = content_rect.x - self.timeline_offset
        for i, segment in enumerate(self.segments):
            segment_width = self._get_segment_width(segment)
            segment_rect = pygame.Rect(
                x_pos, 
                content_rect.y + 10,
                segment_width, 
                self.segment_height
            )
            
            # Skip if completely outside visible area
            if x_pos > content_rect.right or x_pos + segment_width < content_rect.x:
                x_pos += segment_width + self.segment_spacing
                continue
            
            # Determine segment color based on status and type
            segment_color = self.segment_colors.get(segment["status"], self.segment_colors["pending"])
            
            # Adjust brightness if selected or hovered
            if i == self.selected_segment_index:
                # Selected segment: full brightness
                brightness_factor = 1.0
                border_color = (255, 255, 255)
                border_width = 2
            elif i == self.hover_segment_index:
                # Hovered segment: slightly brighter
                brightness_factor = 0.9
                border_color = (200, 200, 220)
                border_width = 1
            else:
                # Normal segment: slightly dimmer
                brightness_factor = 0.7
                border_color = None
                border_width = 0
            
            # Apply brightness factor
            adjusted_color = tuple(min(255, int(c * brightness_factor)) for c in segment_color)
            
            # Draw segment background with rounded corners
            pygame.draw.rect(screen, adjusted_color, segment_rect, border_radius=5)
            
            # Draw border if selected or hovered
            if border_color:
                pygame.draw.rect(screen, border_color, segment_rect, width=border_width, border_radius=5)
            
            # Draw segment label
            label_text = f"Segment {i+1}: {segment['type'].title()}"
            label = self.segment_font.render(label_text, True, (30, 30, 40))
            screen.blit(label, (segment_rect.x + 10, segment_rect.y + 8))
            
            # Draw status badge
            status_text = segment["status"].upper()
            status_color = (240, 240, 240)
            status_label = self.segment_font.render(status_text, True, status_color)
            status_bg_rect = status_label.get_rect(
                bottomright=(segment_rect.right - 8, segment_rect.bottom - 8)
            )
            status_bg_rect.inflate_ip(10, 6)
            
            # Status badge background
            status_bg_color = self.segment_colors.get(segment["status"], (100, 100, 100))
            pygame.draw.rect(screen, status_bg_color, status_bg_rect, border_radius=4)
            
            # Status text
            screen.blit(status_label, (
                status_bg_rect.centerx - status_label.get_width() // 2,
                status_bg_rect.centery - status_label.get_height() // 2
            ))
            
            # Advance x position
            x_pos += segment_width + self.segment_spacing
        
        # Reset clipping
        screen.set_clip(original_clip)
        
        # Draw scroll indicators if needed
        if self.total_width > self.visible_width:
            if self.timeline_offset > 0:
                # Left arrow
                pygame.draw.polygon(screen, (180, 180, 180), [
                    (self.rect.x + 15, self.rect.centery),
                    (self.rect.x + 25, self.rect.centery - 10),
                    (self.rect.x + 25, self.rect.centery + 10)
                ])
            
            if self.timeline_offset < self.total_width - self.visible_width:
                # Right arrow
                pygame.draw.polygon(screen, (180, 180, 180), [
                    (self.rect.right - 15, self.rect.centery),
                    (self.rect.right - 25, self.rect.centery - 10),
                    (self.rect.right - 25, self.rect.centery + 10)
                ])
    
    def update(self, mouse_pos, dt=1/60):
        """Update manager state based on mouse position"""
        # Update animations
        self.pulse_progress = (self.pulse_progress + dt) % 1.0
        
        # Check if mouse is over the content area
        content_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.y + 45,
            self.rect.width - 20,
            self.rect.height - 55
        )
        
        if content_rect.collidepoint(mouse_pos) and self.segments:
            # Find which segment the mouse is hovering over
            x_pos = content_rect.x - self.timeline_offset
            prev_hover = self.hover_segment_index
            self.hover_segment_index = -1
            
            for i, segment in enumerate(self.segments):
                segment_width = self._get_segment_width(segment)
                segment_rect = pygame.Rect(
                    x_pos, 
                    content_rect.y + 10,
                    segment_width, 
                    self.segment_height
                )
                
                if segment_rect.collidepoint(mouse_pos):
                    self.hover_segment_index = i
                    break
                
                x_pos += segment_width + self.segment_spacing
            
            # Trigger hover animation if changed
            if prev_hover != self.hover_segment_index:
                self.hover_alpha = 180  # Start hover glow animation
        else:
            self.hover_segment_index = -1
        
        # Update hover animation
        if self.hover_alpha > 0:
            self.hover_alpha = max(0, self.hover_alpha - 300 * dt)
    
    def handle_event(self, event):
        """Handle mouse events for segment selection and timeline scrolling"""
        content_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.y + 45,
            self.rect.width - 20,
            self.rect.height - 55
        )
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if content_rect.collidepoint(event.pos):
                # Handle scrolling with mouse wheel
                if event.button == 4:  # Scroll up/left
                    self.timeline_offset = max(0, self.timeline_offset - 40)
                    return True
                
                elif event.button == 5:  # Scroll down/right
                    max_offset = max(0, self.total_width - self.visible_width)
                    self.timeline_offset = min(max_offset, self.timeline_offset + 40)
                    return True
                
                # Handle segment selection with left click
                elif event.button == 1:
                    # If hovering over a segment, select it
                    if self.hover_segment_index != -1:
                        old_selection = self.selected_segment_index
                        self.selected_segment_index = self.hover_segment_index
                        
                        # Notify callback if selection changed
                        if old_selection != self.selected_segment_index and self.on_segment_selected:
                            self.on_segment_selected(self.selected_segment_index)
                        
                        return True
                    
                    # Otherwise start dragging the timeline
                    else:
                        self.is_dragging = True
                        self.drag_start_x = event.pos[0]
                        self.drag_start_offset = self.timeline_offset
                        return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # End dragging
            if event.button == 1 and self.is_dragging:
                self.is_dragging = False
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            # Update dragging
            if self.is_dragging:
                dx = self.drag_start_x - event.pos[0]
                new_offset = self.drag_start_offset + dx
                
                # Constrain within bounds
                self.timeline_offset = max(0, min(self.total_width - self.visible_width, new_offset))
                return True
        
        return False
    
    def add_new_segment(self, segment_data, ai_composer):
        """Add a new segment and update the display"""
        ai_composer.add_segment(segment_data["path"], segment_data["type"])
        self.update_segments(ai_composer)
        
    def get_selected_segment(self):
        """Get the currently selected segment"""
        if 0 <= self.selected_segment_index < len(self.segments):
            return self.segments[self.selected_segment_index]
        return None