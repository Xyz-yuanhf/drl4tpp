import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F

class StateUTPP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    # Product-market data
    supply_data: torch.Tensor
    price_data: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    demand_remaining: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        loc = input['loc']
        depot = input['depot']
        supply_data = input['supply_data']
        price_data = input['price_data']

        batch_size, n_loc, _ = loc.size()
        n_product = supply_data.size()[-1]
        coords = torch.cat((depot[:, None, :], loc), -2)
        padding = torch.zeros_like(supply_data[:, 0:1, :])
        return StateUTPP(
            coords=coords,
            supply_data=torch.cat((padding, supply_data), dim=1),
            price_data=torch.cat((padding, price_data), dim=1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 1 + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            demand_remaining=torch.ones(batch_size, n_product, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def update(self, selected):

        assert self.i.size(0) == 1, 'Can only update if state represents single step'

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update the remaining demand
        cur_supply = self.supply_data[self.ids, selected].squeeze()
        demand_remaining = torch.maximum(self.demand_remaining - cur_supply, torch.zeros_like(self.demand_remaining))

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, demand_remaining=demand_remaining, i=self.i + 1
        )

    def get_remaining_demand(self):
        return self.demand_remaining

    def all_finished(self):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        return self.i.item() > 0 and (self.prev_a == 0).all()
        # return self.visited[:, :, 0].all()  # If we have visited the depot we're done

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):
        """
        Forbids to visit depot twice in a row, unless all nodes have been visited
        Depot is feasible to reach if and only if all demands are satisfied
        """
        mask = self.visited > 0
        demand_satisfied = torch.max(self.demand_remaining, dim=-1)[0] > 0

        mask = mask | mask[:, :, 0:1]  # if the depot has already been visited then we cannot visit anymore
        mask[:, :, 0] = demand_satisfied[:, None]  # if feasible to back to depot
        return mask
