"""Group 4 - What kinds of customers do we have?"""
import matplotlib.pyplot as plt
import streamlit as st

from ml_base import load_customers
from w03_group4_segments import CustomerSegments

st.title("🧩 What kinds of customers do we have?")
st.caption("Card 4 · Team: Zuzanna, Jan, Lena")

customers = load_customers()
model = CustomerSegments(customers, k=4, scale=True).fit()
profile = model.profile()                                   # one row per segment, highest mean revenue first

# our names for the 4 segments, in the order of profile()
NAMES = ["🏆 Champions", "😴 Regulars going quiet", "🌱 New & promising", "👋 One-time & gone"]
champions, quiet, new, gone = [profile.iloc[i] for i in range(4)]
segment_revenue = profile["customers"] * profile["revenue"]            # customers x mean revenue = total revenue
revenue_share = (segment_revenue / segment_revenue.sum() * 100).round(0)
customer_share = (profile["customers"] / len(customers) * 100).round(0)

st.header("1️⃣ What we built")
st.markdown(f"""
`CustomerSegments(customers, k=4, scale=True)` sorts our **{len(customers):,} customers** into `k` segments with **k-means**.
It looks at three numbers per customer: days since the last purchase (`recency_days`), number of `orders` and `revenue`.

`.fit()` → the segment number of every customer · `.profile()` → one row per segment · `.plot()` → orders vs revenue, coloured by segment
""")

st.header("2️⃣ Our answer to the board")
col1, col2, col3 = st.columns(3)
col1.metric("Revenue from Champions", f"{revenue_share.iloc[0]:.0f}%")
col2.metric("Champions among customers", f"{customer_share.iloc[0]:.0f}%")
col3.metric("Came back: best vs worst", f"{champions['returned_pct']:.0f}% vs {gone['returned_pct']:.0f}%",
            help=f"Champions vs One-time & gone. All customers: {customers['returned'].mean() * 100:.0f}% bought again between 1 Sep and 9 Dec 2011")

named = profile.copy()
named.insert(0, "our name", NAMES)
named["revenue_pct"] = revenue_share
st.dataframe(named)
st.caption("recency_days, orders, revenue = the averages of the segment · returned_pct = % who bought again after 1 Sep 2011 · "
           "revenue_pct = the segment's share of all revenue")

fig = model.plot().figure
st.pyplot(fig)
plt.close(fig)

st.markdown(f"""
**We have four kinds of customers:**
- **{NAMES[0]}** - ordered {champions['orders']:.0f} times, last time about {champions['recency_days']:.0f} days ago,
  £{champions['revenue']:,.0f} each. {champions['returned_pct']:.0f}% come back anyway: keep them happy, no vouchers needed.
- **{NAMES[1]}** - good customers (£{quiet['revenue']:,.0f} each, {quiet['orders']:.0f} orders), but their last order was
  about {quiet['recency_days']:.0f} days ago and only {quiet['returned_pct']:.0f}% came back.
- **{NAMES[2]}** - bought only {new['orders']:.0f} times, but very recently ({new['recency_days']:.0f} days ago): turn them into regulars.
- **{NAMES[3]}** - one small order (£{gone['revenue']:,.0f}) about {gone['recency_days']:.0f} days ago.
  The biggest group ({customer_share.iloc[3]:.0f}% of customers) but only {revenue_share.iloc[3]:.0f}% of the revenue.

**Who should get attention?** The **Regulars going quiet**: {quiet['customers']:.0f} customers with
{revenue_share.iloc[1]:.0f}% of our revenue, and {100 - quiet['returned_pct']:.0f}% of them did not come back.
A win-back campaign there protects the most money.
""")

st.header("🎛️ Try it yourself")
col1, col2 = st.columns(2)
k = col1.slider("Number of segments (k)", 2, 8, 4)                      # min, max, start value
scale = col2.checkbox("Scale the numbers first", value=True, help="log1p + StandardScaler before k-means")

tried = CustomerSegments(customers, k=k, scale=scale).fit()
tried_profile = tried.profile()
st.dataframe(tried_profile)
fig = tried.plot().figure
st.pyplot(fig)
plt.close(fig)
if not scale:
    biggest = tried_profile["customers"].max() / len(customers) * 100
    st.warning(f"Without scaling, {biggest:.0f}% of all customers land in ONE segment. k-means measures distance in raw "
               "numbers: £1,000 more revenue counts 1,000 times as much as 1 more order, so only revenue decides "
               "and a few giant customers get segments of their own.")

st.header("3️⃣ Is our number real? + what the AI got wrong")
typical_champion = customers.loc[model.results_ == profile.index[0], "revenue"].median()
st.markdown(f"""
- **k = 4 is our choice, not a fact.** k-means always returns exactly `k` groups, also when the customers form one
  continuous cloud (look at the plot: there are no gaps). Move the slider: 3 or 5 segments look just as reasonable.
- **The segments depend on how we scale.** Untick the box above: the same customers, a completely different answer.
- **The return rates are partly built in.** We used `recency_days` to make the segments, and people who bought
  recently come back more often - so the segments *had to* differ in `returned_pct`. And "came back" was measured
  only in Sep - Dec 2011, the Christmas season.
- **Averages hide giants.** The Champions' mean revenue is £{champions['revenue']:,.0f}, but the typical (median)
  Champion spent only £{typical_champion:,.0f}: a few wholesale customers pull the average up.
- **AI:** the first `plot()` the AI wrote drew one segment after the other. Orders are whole numbers, so thousands of
  points sat on exactly the same spots and one whole segment (over 500 customers) was almost completely hidden
  behind the others. We changed it: the points are drawn in random order and shifted a little left/right.
""")
