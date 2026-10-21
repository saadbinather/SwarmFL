/** Preserve the PyData primary sidebar state and its accessible label. */
(() => {
  const storageKey = "swarmfl-sidebar-collapsed";
  const root = document.documentElement;

  try {
    if (window.localStorage.getItem(storageKey) === "true") {
      root.dataset.swarmflSidebar = "collapsed";
    }
  } catch (_error) {
    // Storage can be disabled without affecting the sidebar itself.
  }

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("pst-collapse-sidebar-button");
    const sidebar = document.getElementById("pst-primary-sidebar");
    const mobileClose = document.querySelector(".swarmfl-sidebar-close");
    const mobileDialog = document.getElementById("pst-primary-sidebar-modal");

    if (mobileClose && mobileDialog) {
      mobileClose.addEventListener("click", () => {
        if (mobileDialog.open) {
          mobileDialog.close();
        }
      });
    }

    if (!button || !sidebar) {
      return;
    }

    if (root.dataset.swarmflSidebar === "collapsed") {
      sidebar.classList.add("pst-squeeze");
      button.setAttribute("aria-expanded", "false");
    }

    const syncState = () => {
      const collapsed = button.getAttribute("aria-expanded") === "false";
      const action = collapsed ? "Expand navigation" : "Collapse navigation";
      button.setAttribute("aria-label", action);
      button.setAttribute("title", action);

      if (collapsed) {
        root.dataset.swarmflSidebar = "collapsed";
      } else {
        delete root.dataset.swarmflSidebar;
      }

      try {
        window.localStorage.setItem(storageKey, String(collapsed));
      } catch (_error) {
        // Keep the interaction functional when storage is unavailable.
      }
    };

    new MutationObserver(syncState).observe(button, {
      attributes: true,
      attributeFilter: ["aria-expanded"],
    });
    syncState();
  });
})();
