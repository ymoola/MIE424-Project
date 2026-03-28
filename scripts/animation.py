# Run from the repository root with:
#   manimgl scripts/animation.py Lookahead

from manimlib import *
import numpy as np

class Lookahead(Scene):
    def construct(self):
        # 1. Coordinate System & Title
        axes = Axes(
            x_range=(-5, 5),
            y_range=(-3, 3),
            height=7,
            width=12,
            axis_config={"stroke_color": GREY_B, "stroke_width": 2}
        )
        
        title = Text("Lookahead Optimizer Convergence", font_size=40, color=BLUE_A)
        title.to_edge(UP, buff=0.3)
        
        # 2. Loss Landscape (Contours)
        # We define a 'valley' function: f(x, y) = 0.1*x^2 + y^2
        # We use ImplicitFunction to draw level sets (contours)
        def loss_func(x, y):
            # Map screen coords back to axes coords for math
            coord = axes.p2c([x, y, 0])
            return 0.1 * coord[0]**2 + coord[1]**2

        contours = VGroup(*[
            ImplicitFunction(
                lambda x, y: loss_func(x, y) - val,
                color=TEAL_E,
                stroke_opacity=0.4
            )
            for val in [0.1, 0.5, 1, 2, 4, 8]
        ])
        
        self.add(axes, contours, title)
        self.play(FadeIn(contours, lag_ratio=0.1), Write(title), run_time=3)

        # 3. Weights & Legend
        slow_dot = Dot(axes.c2p(-4, 2), color=RED, radius=0.12)
        fast_dot = slow_dot.copy().set_color(BLUE)
        
        slow_label = Text("Slow Weights (Long-term)", color=RED).scale(0.5)
        fast_label = Text("Fast Weights (Exploration)", color=BLUE).scale(0.5)
        legend = VGroup(fast_label, slow_label).arrange(DOWN, aligned_edge=LEFT).to_corner(UL, buff=1)
        
        # Trace paths to show convergence better
        fast_trace = TracedPath(fast_dot.get_center, stroke_color=BLUE, stroke_width=2, stroke_opacity=0.6)
        slow_trace = TracedPath(slow_dot.get_center, stroke_color=RED, stroke_width=4)
        
        self.add(fast_trace, slow_trace)
        self.play(FadeIn(slow_dot), FadeIn(fast_dot), FadeIn(legend), run_time=2)
        self.wait(1)

        # 4. Optimization Loop (4 Iterations ~ 35 seconds total)
        # Parameters
        k_steps = 5
        alpha = 0.5 # Slow weight step size
        learning_rate = 0.8
        
        # Current state in coordinate space
        curr_slow_coord = np.array([-4.0, 2.0])
        curr_fast_coord = curr_slow_coord.copy()

        for iteration in range(4):
            # --- Fast Steps Phase ---
            step_group = VGroup()
            for i in range(k_steps):
                # Gradient descent step: Move toward (0,0) with some noise/curvature
                # Gradient of 0.1x^2 + y^2 is [0.2x, 2y]
                grad = np.array([0.2 * curr_fast_coord[0], 2.0 * curr_fast_coord[1]])
                curr_fast_coord -= learning_rate * grad
                
                target_pos = axes.c2p(*curr_fast_coord)
                
                # Animate the fast movement
                self.play(
                    fast_dot.animate.move_to(target_pos),
                    run_time=1.2,
                    rate_func=smooth
                )
            
            self.wait(0.5)
            
            # --- Slow Update (Lookahead) Phase ---
            # slow = slow + alpha * (fast - slow)
            new_slow_coord = curr_slow_coord + alpha * (curr_fast_coord - curr_slow_coord)
            new_slow_pos = axes.c2p(*new_slow_coord)
            
            # Draw the interpolation line
            sync_line = DashedLine(slow_dot.get_center(), fast_dot.get_center(), color=YELLOW, stroke_width=2)
            sync_text = Text(f"Lookahead Sync {iteration+1}", font_size=24, color=YELLOW).next_to(sync_line, UP)
            
            self.play(ShowCreation(sync_line), FadeIn(sync_text), run_time=1.5)
            self.play(
                slow_dot.animate.move_to(new_slow_pos),
                # Reset fast weights to the new slow position
                fast_dot.animate.move_to(new_slow_pos),
                run_time=2.5
            )
            
            # Update state
            curr_slow_coord = new_slow_coord
            curr_fast_coord = new_slow_coord.copy()
            
            self.play(FadeOut(sync_line), FadeOut(sync_text), run_time=1)
            self.wait(1)

        # 5. Conclusion
        finish_text = Text("Converged to Local Minimum", font_size=32).to_edge(DOWN, buff=1)
        self.play(Write(finish_text), slow_dot.animate.scale(1.5), run_time=2)
        self.wait(5)
