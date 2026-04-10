import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import math

def visualize_marathon_audit():
    print("Loading Master Marathon Data (Synchronized)...")
    seq_obs = np.load("marathon_sequential_obs.npy")
    sto_obs = np.load("marathon_stochastic_obs.npy")
    
    # Static Audit Figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Sequential 3D Path
    ax1 = fig.add_subplot(221, projection='3d')
    num_agents = seq_obs.shape[0]
    for i in range(min(num_agents, 1)):
        ax1.plot(seq_obs[i, :, 13], seq_obs[i, :, 12], seq_obs[i, :, 14], label='Actual Path')
    ax1.set_title("Sequential Marathon (Synchronized 20-D)")
    ax1.set_xlabel("East (ft)")
    ax1.set_ylabel("North (ft)")
    ax1.set_zlabel("Alt (ft)")
    # Force equal aspect ratio to see the turn clearly
    ax1.set_box_aspect((np.ptp(seq_obs[0,:,13]), np.ptp(seq_obs[0,:,12]), np.ptp(seq_obs[0,:,14]))) 

    # 2. Stochastic 3D Path
    ax2 = fig.add_subplot(222, projection='3d')
    for i in range(min(num_agents, 1)):
        ax2.plot(sto_obs[i, :, 13], sto_obs[i, :, 12], sto_obs[i, :, 14], color='magenta')
    ax2.set_title("Stochastic Gauntlet (100% Difficulty)")
    ax2.set_xlabel("East (ft)")
    ax2.set_ylabel("North (ft)")
    ax2.set_box_aspect((np.ptp(sto_obs[0,:,13]), np.ptp(sto_obs[0,:,12]), np.ptp(sto_obs[0,:,14]))) 
    
    # 3. Bank Angle Audit (Crucial to verify the agent IS turning)
    ax3 = fig.add_subplot(223)
    # obs[0] is phi (roll)
    roll_deg = np.degrees(seq_obs[0, :, 0])
    ax3.plot(roll_deg, color='blue', label='Bank Angle (deg)')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title("Agent Response: Bank Angle (Phi)")
    ax3.set_ylabel("Degrees")
    ax3.legend()
    
    # 4. Heading Trajectory (Top View)
    ax4 = fig.add_subplot(224)
    ax4.plot(seq_obs[0, :, 13], seq_obs[0, :, 12], label='Ground Track')
    ax4.set_title("Top-Down Ground Track (X-Y)")
    ax4.set_xlabel("East deviation (ft)")
    ax4.set_ylabel("Forward Progress (ft)")
    ax4.axis('equal') # VERY IMPORTANT
    
    plt.tight_layout()
    plt.savefig("pioneer_marathon_audit.png")
    print("Static Audit: Saved as pioneer_marathon_audit.png")

    # --- INTERACTIVE DASHBOARD ---
    fig_interactive = go.Figure()
    
    fig_interactive.add_trace(go.Scatter3d(
        x=seq_obs[0, :, 13], y=seq_obs[0, :, 12], z=seq_obs[0, :, 14],
        mode='lines',
        name='Sequential Marathon',
        line=dict(color='cyan', width=4)
    ))
    
    fig_interactive.add_trace(go.Scatter3d(
        x=sto_obs[0, :, 13], y=sto_obs[0, :, 12], z=sto_obs[0, :, 14],
        mode='lines',
        name='Stochastic Gauntlet',
        line=dict(color='magenta', width=4, dash='dash')
    ))
    
    fig_interactive.update_layout(
        title="Pioneer Master Marathon: Coordinated Flight Audit",
        scene=dict(
            xaxis_title='East (ft)',
            yaxis_title='North (ft)',
            zaxis_title='Altitude (ft)',
            aspectmode='data', # THIS FIXES THE STRAIGHT LINE VISUAL
            bgcolor='rgb(10, 10, 10)'
        ),
        template="plotly_dark"
    )
    fig_interactive.write_html("pioneer_marathon_playback.html")
    print("Interactive Dashboard: Saved as pioneer_marathon_playback.html")

if __name__ == "__main__":
    visualize_marathon_audit()
