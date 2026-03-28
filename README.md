<div align="center">

# 🎵 TikTok-Style React Video Player

**A high-fidelity, mobile-first short-form video experience built for the modern web.**

<br />

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)](https://vitejs.dev/)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

> Building the future of mobile-first scrolling with raw HTML5 video power and butter-smooth CSS snap physics!

</div>

<br/>

## 🍿 The Experience

<p align="center">
  <a href="https://drive.google.com/file/d/1mbvV-_ll-07wa-aQlp09GrB7QoRaJ8-l/view?usp=drive_link">
    <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="60" alt="Watch the Demo" />
  </a>
  <br />
  <b>Tap the icon above to watch the Full Walkthrough</b>
</p>

---

<table align="center" style="width: 100%; border: none;">
  <tr>
    <td width="50%" valign="top">
      <h3>⚡ Lightning Fast Core</h3>
      <ul>
        <li><b>Scroll Snap:</b> Silky smooth <code>&lt;CSS scroll-snap&gt;</code> integration for native-feeling vertical swiping.</li>
        <li><b>Infinite Loop Illusion:</b> Seamlessly loops boundaries <code>[last...first]</code> via invisible coordinate leaps!</li>
        <li><b>IntersectionObserver:</b> Playback is paused automatically outside the viewport entirely saving CPU/Battery life.</li>
        <li><b>Loading Skeletons:</b> Beautiful shimmer effects mask the layout gracefully while heavy video assets buffer.</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🎭 Rich Overlays & Gestures</h3>
      <ul>
        <li><b>Double Tap:</b> Triggers a gorgeous animated center-screen heart pop.</li>
        <li><b>Long Press to Pause:</b> Instantly suspends the video while the screen is held.</li>
        <li><b>Music Engine:</b> An elegant spinning vinyl record reacting exactly to the playback state.</li>
        <li><b>Dual Theme Modes:</b> Included universal light and dark modes in a sleek settings popup to change the vibe instantly.</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🛠️ The Technology Behind The Magic

<details>
  <summary><b>✨ View System Architecture & Hierarchy</b></summary>
  
  ```mermaid
  graph TD;
      A[App Shell & Theme Provider] --> B[Infinite Scroll Feed UI];
      B --> C[Video Card Context];
      C --> D[HTML5 Video Controller];
      C --> E[Right Action Bar Overlay];
      C --> F[Bottom User Info Panel];
      C --> G[Spinning Music Disc];
  ```

  ### 💡 Intelligent Video Management
  Using `useRef` directly on `<video>` nodes instead of hammering React state `onTimeUpdate` completely eliminates massive re-renders. Every single play/pause action and progress bar update feels incredibly responsive.
</details>

<details>
  <summary><b>📁 View Project Folder Tree</b></summary>

  ```text
  tiktok-player-khushipriyadarshni/
  ├── src/
  │   ├── components/
  │   │   ├── VideoCard.jsx      # Auto-play / Heavy Gesture Logic
  │   │   ├── ActionBar.jsx      # Likes, Comments, Saves
  │   │   └── UserInfo.jsx       # Avatar & Follow System Toggle
  │   ├── data/
  │   │   └── videos.js          # Video segment boundaries & mock backend
  │   └── App.jsx                # Feed generation and Infinite logic
  ├── index.html
  └── package.json
  ```
</details>


## 🚀 Boot It Up

```bash
# 1. Grab the code
git clone https://github.com/khushipriyadarshni/tiktok-player-khushipriyadarshni.git

# 2. Drop into the directory
cd tiktok-player-khushipriyadarshni

# 3. Install packages & fly!
npm install && npm run dev
```

Visit [`http://localhost:5173`](http://localhost:5173) and resize your browser window or open your mobile phone simulator to experience the complete responsive design!

<div align="center">

<br/><hr/><br/>

### Built with ❤️ by **Khushi Priyadarshni**
*B.E. Artificial Intelligence & Data Science*

**Thank you for exploring my project!**
</div>
