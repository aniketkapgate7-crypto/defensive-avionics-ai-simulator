"""Minimal Pygame starter window for the deterministic synthetic demo."""

from __future__ import annotations

from defensive_avionics.integration.orchestrator import DemoOrchestrator


def run() -> int:
    try:
        import pygame
    except ImportError:
        print('Pygame is not installed. Run: pip install -e ".[ui]"')
        return 1

    pygame.init()
    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Multi-Modal AI Simulator — Academic Demo")
    font = pygame.font.SysFont("consolas", 26)
    clock = pygame.time.Clock()
    frames = DemoOrchestrator().synthetic_sequence()
    frame_index = 0
    elapsed = 0
    running = True

    while running:
        delta_ms = clock.tick(30)
        elapsed += delta_ms
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if elapsed >= 1000:
            frame_index = (frame_index + 1) % len(frames)
            elapsed = 0

        snapshot = frames[frame_index]
        screen.fill((5, 12, 20))
        lines = [
            "OFFLINE ACADEMIC SIMULATION",
            f"SYSTEM STATUS: {snapshot.status.upper()}",
            f"SIGNAL CLASS: {snapshot.signal_label}",
            f"SIGNAL CONFIDENCE: {snapshot.signal_confidence:.0%}",
            f"RELATIVE URGENCY: {snapshot.vision_urgency.upper()}",
            f"ABSTRACT POLICY: {snapshot.policy_action}",
        ]
        for index, line in enumerate(lines):
            color = (30, 230, 190) if index == 0 else (225, 240, 245)
            screen.blit(font.render(line, True, color), (55, 55 + index * 72))
        pygame.display.flip()

    pygame.quit()
    return 0
