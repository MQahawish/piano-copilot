import pygame
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
    