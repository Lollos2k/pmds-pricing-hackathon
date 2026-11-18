"""Inventory-aware bidding and adaptive retail pricing for Dynamic Market Exchange."""
from collections import Counter
from math import ceil, floor

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing


class AdaptiveMarket(BaseAgent):
    opening_bid = 1.85
    opening_retail = 5.2
    retail_spread = 0.20

    def __init__(self):
        self.identity = None
        self.auction_cursor = 0
        self.sales_cursor = 0
        self.bid_ratios = {}
        self.needed = {}
        self.prices = {}
        self.volumes = {}
        self.offered = {}
        self.quiet_turns = Counter()
        self.opening_set = False
        self.tier_sales = {}
        self.spread_by_good = {}
        self.premium_stockouts = Counter()
        self.recent_revenue = {}
        self.recent_sellthrough = {}
        self.revenue_samples = Counter()
        self.expansion_streak = Counter()

    def _transactions(self, state):
        if self.identity is None:
            matches = [team for team, cash in state.public_capital.items()
                       if cash == state.capital and state.public_inventories[team] == state.inventory]
            if len(matches) == 1:
                self.identity = matches[0]
        wins = Counter()
        sales = Counter()
        for tx in state.auction_transactions[self.auction_cursor:]:
            if tx.buyer == self.identity:
                wins[tx.good] += tx.quantity
        for tx in state.sales_transactions[self.sales_cursor:]:
            if tx.seller == self.identity:
                sales[tx.good] += tx.quantity
        self.auction_cursor = len(state.auction_transactions)
        self.sales_cursor = len(state.sales_transactions)
        return wins, sales

    def decide(self, state, goods_config):
        solvent_teams = sum(cash > 0 for cash in state.public_capital.values())
        protect_margin = solvent_teams >= 12
        transactions = state.sales_transactions[self.sales_cursor:]
        wins, sales = self._transactions(state)
        lots = {lot.good: lot for lot in state.auction_lots}
        if not self.opening_set:
            self.opening_set = True
            firms = max(1, len(state.public_capital))
            basket = Counter()
            for recipe in goods_config.recipes.values():
                basket.update(recipe.inputs)
            room = min((lots[m].quantity if m in lots else 0) / units
                       for m, units in basket.items()) / firms
            if len(goods_config.finished_goods) == 1 and room < 8:
                self.opening_bid = 1.44
                self.opening_retail = 3.55
            else:
                scarcity = min(1.0, max(0.0, (2.2 - room) / 1.4))
                crowd = min(1.0, max(0.0, (firms - 3) / 2))
                self.opening_bid += 0.63 * scarcity * crowd
                self.opening_retail += 2.55 * scarcity * crowd
        sales_by_price = Counter()
        revenue = Counter()
        if self.identity is not None:
            for sale in transactions:
                if sale.seller == self.identity:
                    sales_by_price[sale.good, sale.price] += sale.quantity
                    revenue[sale.good] += sale.quantity * sale.price
        quotes = {}
        for material, lot in lots.items():
            ratio = self.bid_ratios.get(material, self.opening_bid)
            wanted = self.needed.get(material, 0)
            if wanted and self.identity is not None:
                fill = wins[material] / wanted
                if fill < 0.92:
                    ratio = min(8.0, ratio * (1.10 + 0.30 * (1 - fill)))
                else:
                    ratio = max(1.015, ratio * 0.961)
            if protect_margin and not wanted and material in self.bid_ratios:
                ratio = max(1.015, ratio * .963)
            self.bid_ratios[material] = ratio
            # A small opening premium gives priority over identical bids.
            opening_multiplier = 1.004 if state.turn <= 2 else 1.0
            bid = lot.reserve_price * ratio * opening_multiplier
            if lot.reserve_price == 0:
                estimated = min((price for price in self.prices.values() if price > 0), default=1)
                bid = estimated * .0015 * ratio
            quotes[material] = bid

        # Several recipes can make the same good. Keep one feasible alternative
        # per good to avoid counting the same demand more than once.
        alternatives = {}
        for name, recipe in goods_config.recipes.items():
            feasible = min((state.inventory.get(m, 0) + (lots[m].quantity if m in lots else 0)) // n
                           for m, n in recipe.inputs.items())
            cost = sum(n * quotes.get(m, 0.0) for m, n in recipe.inputs.items())
            score = (feasible == 0, cost, -feasible, name)
            good = recipe.output_good
            if good not in alternatives or score < alternatives[good][0]:
                alternatives[good] = (score, name, recipe, feasible, cost)

        ranking = []
        can_expand = False
        replacement_costs = {}
        team_count = max(1, len(state.public_capital))
        for good, (_, name, recipe, feasible, cost) in alternatives.items():
            stock = state.inventory.get(good, 0)
            base_cost = sum(n * lots[m].reserve_price for m, n in recipe.inputs.items() if m in lots)
            anchor = max(1.0, self.opening_retail * base_cost)
            # When a single product can be built in bulk, the supply lot is a
            # weak proxy for retail demand. Probe the market before committing.
            bulk_market = len(alternatives) == 1 and feasible >= 32 * team_count
            if bulk_market:
                anchor = max(anchor, min(20 * max(base_cost, 1), state.capital / 40))
            price = self.prices.get(good, anchor)
            demand = self.volumes.get(good)
            if demand is None:
                initial_share = feasible / (max(1, len(alternatives)) ** 0.65 * team_count ** 0.36)
                demand = min(28.0 if bulk_market else 300.0, max(2.0, initial_share))
            sold = sales[good]
            actual = min(self.offered.get(good, 0), sold + stock)
            ready = False
            sold_out = False
            if actual and self.identity is not None:
                exposed_fraction = sold / actual
                old = self.recent_sellthrough.get(good, exposed_fraction)
                self.recent_sellthrough[good] = .62 * old + .38 * exposed_fraction
                if sold:
                    realized = revenue[good] / sold
                    old = self.recent_revenue.get(good, realized)
                    self.recent_revenue[good] = .64 * old + .36 * realized
                    self.revenue_samples[good] += sold
                sold_out = sold >= 2 and exposed_fraction >= .92 and actual >= .80 * self.offered.get(good, 0)
                if sold_out and revenue[good] > 1.28 * cost * sold:
                    self.expansion_streak[good] += 1
                else:
                    self.expansion_streak[good] = 0
                # Expand only after repeated sell-through at a positive margin,
                # with enough cash for several turns of replacement inventory.
                ready = (
                    not protect_margin
                    and self.expansion_streak[good] >= 2
                    and state.capital > 7 * cost * (demand + 1)
                )
                can_expand = can_expand or ready
            if actual and self.identity is not None:
                if sold == actual:
                    price *= 1.095
                    scale = 1.52 if ready else 1.32
                    demand = max(demand * .96, scale * sold + 1.0)
                else:
                    fraction = sold / actual
                    demand = 0.53 * demand + 0.47 * (1.12 * sold + 0.55)
                    if fraction < 0.37:
                        price *= 0.82 if sold == 0 else 0.90
                    elif fraction < 0.72:
                        price *= 0.967
                    elif fraction > 0.93:
                        price *= 1.025
                self.quiet_turns[good] = self.quiet_turns[good] + 1 if sold == 0 else 0
            t = self.tier_sales.get(good)
            if t and actual >= self.offered.get(good, 0) and sold == actual:
                _, premium, cheap_count = t
                if sales_by_price[good, premium] >= 3 and actual - cheap_count >= 3:
                    self.premium_stockouts[good] += 1
                    price *= 1.11 if self.premium_stockouts[good] == 1 else 1.19
                else:
                    self.premium_stockouts[good] = 0
            else:
                self.premium_stockouts[good] = 0
            if good in self.tier_sales and self.identity is not None:
                cheap, premium, cheap_units = self.tier_sales[good]
                low_buys = sales_by_price[good, cheap]
                high_buys = sales_by_price[good, premium]
                high_exposed = max(0, actual - cheap_units)
                spread = self.spread_by_good.get(good, self.retail_spread)
                if high_exposed >= 2 and low_buys >= .85 * cheap_units:
                    if high_buys <= .35 * high_exposed:
                        spread = max(.06, spread - .035)
                    elif high_buys >= .9 * high_exposed:
                        spread = min(.25, spread + .015)
                self.spread_by_good[good] = spread
            self.prices[good] = price = max(0.01, price)
            self.volumes[good] = demand = max(0.65, demand)
            goal = max(0, ceil(demand - stock))
            margin = price - cost
            if margin <= 0 and stock == 0 and state.turn % 4 == 0:
                # Recheck a product after a prolonged supply-price spike.
                self.prices[good] = max(price, 1.2 * cost)
            target = min(goal, feasible) if margin > 0 else 0
            if self.quiet_turns[good] >= 2:
                target = min(target, 1)
            if protect_margin and self.revenue_samples[good] >= 3 and good in self.recent_sellthrough:
                sellthrough = self.recent_sellthrough[good]
                observed_price = self.recent_revenue[good]
                expected_price = min(price, .67 * observed_price + .33 * price)
                expected_cash = expected_price * (.52 + .48 * sellthrough)
                if expected_cash < 1.065 * cost:
                    target = min(target, 1) if stock == 0 and state.turn % 4 == 0 else 0
                elif sellthrough < .65 and stock > 0:
                    target = min(target, max(0, ceil(demand * .67) - stock))
                margin = expected_cash - cost
            replacement_costs[good] = cost
            priority = margin / max(1.0, cost ** 0.45)
            ranking.append((priority, name, good, recipe, target))

        raw = dict(state.inventory)
        available = {m:lot.quantity for m,lot in lots.items()}
        budget = (.9 if can_expand else .75) * state.capital
        wanted = Counter()
        assembly = []
        listings = []
        self.offered = {}
        self.tier_sales = {}
        for _, name, good, recipe, target in sorted(ranking, reverse=True):
            quantity = target
            for m,n in recipe.inputs.items():
                quantity = min(quantity, (raw.get(m,0) + available.get(m,0)) // n)

            def spend(units):
                return sum(max(0, units * n - raw.get(m, 0)) * quotes.get(m, 0.0)
                           for m, n in recipe.inputs.items())

            if spend(quantity) > budget:
                low, high = 0, quantity
                while low < high:
                    middle = (low + high + 1) // 2
                    if spend(middle) <= budget:
                        low = middle
                    else:
                        high = middle - 1
                quantity = low
            budget -= spend(quantity)
            if quantity:
                assembly.append(AssemblyOrder(name, quantity))
                for m,n in recipe.inputs.items():
                    required = n * quantity
                    bought = max(0, required - raw.get(m,0))
                    wanted[m] += bought
                    available[m] = available.get(m, 0) - bought
                    raw[m] = max(0, raw.get(m, 0) - required)

            listed = state.inventory.get(good, 0) + quantity
            self.offered[good] = listed
            price = self.prices[good]
            if listed < 3 or self.retail_spread == 0:
                if listed:
                    listings.append(Listing(good, listed, price))
            else:
                lower_quantity = max(1, floor(listed * 0.45))
                spread = self.spread_by_good.get(good, self.retail_spread)
                cheap, premium = price * (1 - spread), price * (1 + spread)
                if protect_margin and quantity and self.revenue_samples[good] >= 3:
                    cheap = max(cheap, replacement_costs[good] * 1.025)
                    premium = max(premium, cheap * 1.002)
                self.tier_sales[good] = (cheap, premium, lower_quantity)
                listings.append(Listing(good, lower_quantity, cheap))
                listings.append(Listing(good, listed - lower_quantity, premium))
        self.needed = wanted
        bids = [Bid(m, qty, quotes[m]) for m, qty in wanted.items() if qty > 0]
        return Decision(bids, assembly, listings)
