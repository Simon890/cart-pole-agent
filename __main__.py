from agent import Agent

if __name__ == "__main__":
    agent = Agent()
    agent.load_model("agent.keras")
    agent.play()
    # agent.train_model(n_max_steps=200, n_iterations=120, n_matches_per_update=15, discount_factor=0.98)
    # agent.save_model("test.keras")