# bot.py
import discord
from discord.ext import commands
from riskgraph import generate_all_risk_plots
import io

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")

@bot.command()
async def riskgraph(ctx):
    """Generate and send the full risk metric chart set."""
    await ctx.send("📊 Generating risk metric plots...")

    try:
        plots = generate_all_risk_plots()

        # Convert each Plotly figure to PNG using kaleido
        for name, fig in plots.items():
            buf = io.BytesIO()
            fig.write_image(buf, format='png')
            buf.seek(0)
            await ctx.send(file=discord.File(buf, filename=f"{name}.png"))

        await ctx.send("✅ All risk plots sent.")

    except Exception as e:
        await ctx.send(f"Failed to generate plots: {e}")
        raise e  # Optional: for local debugging

# Run the bot with your actual token
bot.run("")

