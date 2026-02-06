The latest main branch already includes the fixed‑node world design and the Phase 6/Phase 7 cognitive pipeline. Initial resources are now spawned at fixed positions and only respawn in place after depletion, and the agent’s heuristic has been updated to search for food earlier (energy < 0.75) and water earlier (hydration < 0.4). Craft nodes respawn slowly, hazard nodes are permanent, and the agent’s metabolic drain has been reduced to allow more time for planning. The LLM integration passes an AgentState with energy, hydration, curvature, hazard and food flags, injecting visceral prompts and token biases based on physiological state.

Remaining issues

Hazards are still a constant 3 % of the grid and cannot be cleared. They often trap the agent and contribute little to learning.

The heuristic still includes crafting decisions (it will craft whenever two items are in inventory), even though crafting is supposed to be a high‑level choice best handled by the LLM.

Water search is still relatively late. Hydration falls quickly, so waiting until hydration < 0.4 before looking for water can cause avoidable deaths.

Crafting benefits are not tied to learning signals. Items like flint and baskets provide buffs, but the policy has no explicit reward incentive for crafting them, making it unlikely the learned policy will learn to craft.

Detailed plan for Claude Code

Reduce or disable hazard spawning (simplify the environment):

In grid.py’s _spawn_initial_objects, set the hazard coverage to a much smaller value (e.g. 0.01) or temporarily remove hazards by not spawning any objects of type Hazard. Hazards add difficulty without meaningful reward and prevent the agent from exploring.

Alternatively, implement a hazard‑clearing tool. Define a new tool in tools/builtins.py called clear_hazard that checks for a Hazard at the current position and removes it at a small energy cost. Register the new tool in core.py’s _register_tools. Update the heuristic to call clear_hazard when standing on a hazard instead of just fleeing. This makes hazards an obstacle the agent can learn to deal with.

Delegate crafting to the LLM (remove from heuristic):

Delete the crafting branch in _heuristic_decide (len(self.inventory) >= 2 block) so the heuristic never crafts on its own. Crafting is a strategic decision and should be suggested by the LLM’s reason_plan.

In OllamaReasoner.reason and reason_plan, add a short paragraph to the system prompt listing the available craft recipes (CRAFT_RECIPES) and emphasising that crafting can improve survival (e.g. “Flint improves food energy, Basket increases inventory, Shelter frame reduces night energy cost”). Pass the agent’s inventory in llm_args. This gives the LLM enough information to decide when to craft.

Optionally, bias the LLM’s logits towards the “craft” tool when the agent has craftable items. In _compute_logit_bias, if agent_state.inventory contains two craft items that match a recipe, add a positive bias for the token corresponding to the “craft” tool.

Raise the water‑seeking threshold:

In _heuristic_decide, change the hydration check from self.hydration < 0.4 to a higher threshold, such as 0.6. Adjust the search radius accordingly. This ensures the agent starts looking for water before hydration becomes critical.

Update the emergency hydration branch to move toward water earlier and to prioritise drinking when hydration is below, say, 70 %.

Incorporate crafting benefits into the reward function:

In _compute_reward, add a modest positive reward (e.g. +0.5) when the agent crafts useful items such as flint, basket or shelter_frame. This ties crafting directly into the RL signal and encourages the policy to learn from the LLM’s craft suggestions.

Likewise, consider a small bonus when the agent plants a seed and harvests a crop, to incentivise longer‑term planning.

Optional: further reduce metabolic drain or hazard damage if you still observe starvation or frequent damage despite the above changes. For instance, lower the baseline energy drain to 0.0015 and hydration drain to 0.0006 in tick(), and reduce Hazard.damage in objects.py to lessen the penalty when the agent hits a hazard.

Implementing these modifications will simplify the environment, make crafting a deliberate high‑level action managed by the LLM, and give the policy clearer signals about the value of crafting and hydration management. Once these changes are in place and tested, you can revisit macro‑actions or social extensions and, later, assemble the research narrative.