import pygame

class Button:

    def __init__(self, rect, text, colour=(0,0,0)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.colour = colour

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.colour, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        label = font.render(self.text, True, (0, 0, 0))
        screen.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)