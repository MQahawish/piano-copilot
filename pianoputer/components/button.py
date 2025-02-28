import pygame
import math 
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
