import pygame
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
