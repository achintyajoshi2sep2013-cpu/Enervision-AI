
document.addEventListener("DOMContentLoaded", () => {
  const $ = id => document.getElementById(id);

  const networksBox = $("wifiNetworks");
  const refreshBtn = $("refreshNetworksBtn");
  const wifiStatus = $("wifiStatus");
  const permissionBtn = $("requestPermissionBtn");
  const esp32Status = $("esp32Status");
  const esp32Url = $("esp32Url");
  const sendBtn = $("sendDataBtn");

  let permissionAccepted = false;
  let acceptedUrl = "";

  async function readResponse(response) {
    let data = {};
    try {
      data = await response.json();
    } catch (_) {}

    if (!response.ok || data.success === false) {
      throw new Error(
        data.message || data.error || `HTTP ${response.status}`
      );
    }

    return data;
  }

  async function scanNetworks() {
    networksBox.replaceChildren();
    wifiStatus.textContent = "Scanning Wi-Fi networks...";
    refreshBtn.disabled = true;

    try {
      const response = await fetch("/api/wifi-networks");
      const data = await readResponse(response);

      if (!data.networks?.length) {
        wifiStatus.textContent = data.message || "No networks found.";
        return;
      }

      data.networks.forEach(network => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = network.ssid;
        button.addEventListener("click", () => connectToNetwork(network.ssid));
        networksBox.appendChild(button);
      });

      wifiStatus.textContent = data.message || "Select a network.";
    } catch (error) {
      wifiStatus.textContent = error.message;
    } finally {
      refreshBtn.disabled = false;
    }
  }

  async function connectToNetwork(ssid) {
    const password = window.prompt(
      `Enter the password for "${ssid}". Leave blank only for an open network:`
    );

    if (password === null) return;

    wifiStatus.textContent = `Asking Windows to connect to "${ssid}"...`;

    try {
      const response = await fetch("/api/connect-wifi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ssid, password })
      });

      const data = await readResponse(response);
      wifiStatus.textContent = data.message || "Connection requested.";
    } catch (error) {
      wifiStatus.textContent = error.message;
    }
  }

  permissionBtn.addEventListener("click", async () => {
    permissionAccepted = false;
    acceptedUrl = "";
    sendBtn.disabled = true;

    const url = esp32Url.value.trim();
    esp32Status.textContent = "Requesting ESP32 permission...";

    try {
      const response = await fetch("/api/connection-request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ esp32_url: url })
      });

      const data = await readResponse(response);

      permissionAccepted = data.accepted === true;

      if (permissionAccepted) acceptedUrl = url;

      esp32Status.textContent = data.message || "Permission response received.";

      sendBtn.disabled =
        !permissionAccepted || !window.getLatestApplianceData?.();
    } catch (error) {
      esp32Status.textContent = error.message;
    }
  });

  sendBtn.addEventListener("click", async () => {
    if (!permissionAccepted) {
      esp32Status.textContent = "Request permission first.";
      return;
    }

    const applianceData = window.getLatestApplianceData?.();

    if (!applianceData) {
      esp32Status.textContent = "Detect an appliance first.";
      return;
    }

    sendBtn.disabled = true;
    esp32Status.textContent = "Sending appliance estimate...";

    try {
      const response = await fetch("/api/send-to-esp32", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...applianceData,
          esp32_url: acceptedUrl
        })
      });

      const data = await readResponse(response);
      esp32Status.textContent = data.message || "Transfer completed.";
    } catch (error) {
      esp32Status.textContent = error.message;
    } finally {
      sendBtn.disabled =
        !permissionAccepted || !window.getLatestApplianceData?.();
    }
  });

  refreshBtn.addEventListener("click", scanNetworks);
});
