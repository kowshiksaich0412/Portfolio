const sections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll(".nav a")];
const revealItems = [...document.querySelectorAll(".reveal")];

const navObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      const activeId = `#${entry.target.id}`;
      navLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === activeId);
      });
    });
  },
  {
    rootMargin: "-35% 0px -45% 0px",
    threshold: 0.1,
  }
);

sections.forEach((section) => navObserver.observe(section));

const revealObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    });
  },
  {
    threshold: 0.14,
  }
);

revealItems.forEach((item, index) => {
  item.style.transitionDelay = `${Math.min(index * 90, 420)}ms`;
  revealObserver.observe(item);
});
