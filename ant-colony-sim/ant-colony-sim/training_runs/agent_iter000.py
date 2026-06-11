class MyAgent(AntAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Any persistent initialization can go here

    def decide(self, perception: Perception) -> Action:
        # 1. Automatic interaction with current cell
        if perception.carries:
            if perception.current_is_nest:
                return DropAction(perception.current_x, perception.current_y)
        else:
            if perception.current_has_food:
                return PickUpAction(perception.current_x, perception.current_y)

        # 2. Identify accessible neighbors
        valid_neighbors = [n for n in perception.neighbors if n.is_accessible]
        if not valid_neighbors:
            # Fallback if trapped
            return MoveAction(perception.current_x, perception.current_y)

        # 3. Check for immediate adjacent goals (1-step lookahead)
        if perception.carries:
            for n in valid_neighbors:
                if n.is_nest:
                    return MoveAction(n.x, n.y)
        else:
            for n in valid_neighbors:
                if n.has_food:
                    return MoveAction(n.x, n.y)

        # 4. Filter out recently visited cells to prevent loops
        memory_set = {(pos.x, pos.y) for pos in self.memory}
        unvisited = [n for n in valid_neighbors if (n.x, n.y) not in memory_set]
        
        # If all valid neighbors are in memory (e.g., dead end), fallback to all valid
        candidates = unvisited if unvisited else valid_neighbors

        # 5. Score candidates based on pheromones and crowding
        weights = []
        for n in candidates:
            # Base exploration weight
            weight = 1.0 
            
            if perception.carries:
                # Returning: heavily favor NEST pheromones
                weight += n.nest_pheromone * 15.0
            else:
                # Foraging: heavily favor FOOD pheromones
                weight += n.food_pheromone * 15.0
            
            # Penalize cells with other agents to encourage dispersion and reduce traffic
            weight /= (1.0 + n.agent_count * 2.0)
            
            weights.append(weight)

        # 6. Select move using roulette wheel (weighted random choice)
        total_weight = sum(weights)
        if total_weight > 0:
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            for neighbor, weight in zip(candidates, weights):
                cumulative += weight
                if r <= cumulative:
                    return MoveAction(neighbor.x, neighbor.y)
        
        # Absolute fallback
        chosen = random.choice(candidates)
        return MoveAction(chosen.x, chosen.y)