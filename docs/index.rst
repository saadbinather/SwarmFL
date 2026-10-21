
.. title:: SwarmFL

.. container:: swarmfl-hero

   .. container:: swarmfl-hero-text

      .. raw:: html

         <h1 class="visually-hidden">SwarmFL</h1>
         <img
           class="swarmfl-hero-logo"
           src="_static/swarmfl-logo-transparent.png"
           alt="SwarmFL"
           width="900"
           height="300"
           decoding="async"
         >

      .. container:: swarmfl-intro

         SwarmFL is an open-source Python framework that allows AI specialists
         to collaborate through federated learning. Built on Flower, it helps
         select suitable specialists, combine their predictions, manage
         consultation costs, and improve collaboration using feedback.

      .. code-block:: console
         :class: swarmfl-install

         pip install swarmfl

      .. container:: swarmfl-actions

         - :doc:`Get started <installation>`
         - `View on GitHub <https://github.com/saadbinather/SwarmFL>`_

   .. raw:: html

      <div class="swarmfl-constellation" aria-hidden="true">
        <svg viewBox="0 0 520 440" focusable="false">
          <g class="swarmfl-links">
            <path d="M260 220 L120 110"></path>
            <path d="M260 220 L410 96"></path>
            <path d="M260 220 L440 280"></path>
            <path d="M260 220 L300 390"></path>
            <path d="M260 220 L96 320"></path>
            <path class="swarmfl-link-faint" d="M120 110 L410 96"></path>
            <path class="swarmfl-link-faint" d="M440 280 L300 390"></path>
            <path class="swarmfl-link-faint" d="M96 320 L120 110"></path>
          </g>
          <circle class="swarmfl-halo" cx="260" cy="220" r="64"></circle>
          <circle class="swarmfl-node swarmfl-node--hub" cx="260" cy="220" r="22"></circle>
          <circle class="swarmfl-node" cx="120" cy="110" r="12" style="animation-delay:.4s"></circle>
          <circle class="swarmfl-node" cx="410" cy="96" r="14" style="animation-delay:1.1s"></circle>
          <circle class="swarmfl-node" cx="440" cy="280" r="11" style="animation-delay:1.8s"></circle>
          <circle class="swarmfl-node" cx="300" cy="390" r="13" style="animation-delay:.8s"></circle>
          <circle class="swarmfl-node" cx="96" cy="320" r="10" style="animation-delay:2.2s"></circle>
          <g class="swarmfl-labels">
            <text x="260" y="268">Broker</text>
            <text x="120" y="84">Specialist</text>
            <text x="410" y="66">Specialist</text>
            <text x="440" y="312">Specialist</text>
            <text x="300" y="424">Specialist</text>
            <text x="96" y="352">Specialist</text>
          </g>
        </svg>
      </div>

.. rubric:: Explore the docs
   :class: swarmfl-section-label

.. container:: swarmfl-cards

   - :doc:`Installation <installation>`

     Install from PyPI or from source on Python 3.11–3.13.

   - :doc:`Quickstart <quickstart>`

     Run the synthetic CPU demonstration with three Flower clients.

   - :doc:`Examples <examples/index>`

     The CPU quickstart and a custom specialist walkthrough.

   - :doc:`API Reference <api>`

     Public modules exported from ``fed_med_agents``.

.. rubric:: How SwarmFL fits together
   :class: swarmfl-section-label

.. container:: swarmfl-arch

   .. raw:: html

      <div class="swarmfl-stack" aria-hidden="true">
        <div class="swarmfl-stack__inner">
          <div class="swarmfl-plate swarmfl-plate--1"></div>
          <div class="swarmfl-plate swarmfl-plate--2"></div>
          <div class="swarmfl-plate swarmfl-plate--3"></div>
          <div class="swarmfl-plate swarmfl-plate--4"></div>
          <div class="swarmfl-plate swarmfl-plate--5"></div>
        </div>
      </div>

   .. container:: swarmfl-layers

      .. container:: swarmfl-layer swarmfl-layer-1

         **Specialist registry**

         Compatibility and authorization contracts.

         - :doc:`api/registry`

      .. container:: swarmfl-layer swarmfl-layer-2

         **Routing**

         Fixed, learned, or graph-inspired policies under consultation costs.

         - :doc:`api/routing`
         - :doc:`api/graph-routing`

      .. container:: swarmfl-layer swarmfl-layer-3

         **Consultation**

         Selected-only consultations through Flower.

         - :doc:`api/broker`

      .. container:: swarmfl-layer swarmfl-layer-4

         **Fusion**

         Mean, reliability-weighted, or DeepSets fusion.

         - :doc:`api/dynamic-fusion`
         - :doc:`api/fusion-network`

      .. container:: swarmfl-layer swarmfl-layer-5

         **Feedback and versions**

         Authenticated delayed feedback and validated policy versions.

         - :doc:`api/feedback-events`
         - :doc:`api/continual-feedback`
         - :doc:`api/policy-versions`

.. rubric:: Features
   :class: swarmfl-section-label

.. container:: swarmfl-features

   - **Specialist Registry** Register heterogeneous specialists with explicit
     compatibility and authorization contracts.
   - **Intelligent Routing** Select eligible specialists using fixed, learned,
     or graph-inspired routing policies under consultation costs.
   - **Federated Collaboration** Execute selected-only consultations using
     Flower.
   - **Dynamic Fusion** Combine predictions through mean, reliability-weighted,
     or DeepSets fusion.
   - **Feedback Learning** Support authenticated delayed feedback and federated
     shared-parameter updates.
   - **Version Control** Validate, activate, reject, and roll back learned
     policies.

.. container:: swarmfl-maintainer

   Actively developed and maintained by
   `Saad <https://github.com/saadbinather>`_, `Saif <https://github.com/rajasaifr>`_
   , `Ali <https://github.com/Alihassan0155>`_

.. toctree::
   :maxdepth: 3
   :class: swarmfl-root-nav

   Home <self>
   installation
   quickstart
   examples/index
   api
   research/index
