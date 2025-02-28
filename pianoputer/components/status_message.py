import pygame
import time
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
