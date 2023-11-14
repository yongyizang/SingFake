document.addEventListener('DOMContentLoaded', () => {
  var elems = document.querySelectorAll('.autocomplete');
  var instances = M.Autocomplete.init(elems, {
    data: {
      "Mandarin": null,
      "Cantonese": null,
      "Japanese": null,
      "Korean": null,
      "English": null
    }
  });
  const launchButton = document.getElementById('submitButton');
  launchButton.addEventListener('click', handleButtonClick);
});

var activeTabUrl = '';
var activeTabTitle = '';
var docCount = -1;

const urlField = document.getElementById('url');
const titleField = document.getElementById('title');
const platformField = document.getElementById('platform');
const mainField = document.getElementById('container');

async function getCurrentTab() {
  let queryOptions = { active: true, lastFocusedWindow: true };
  let [tab] = await chrome.tabs.query(queryOptions);
  return tab;
}

function getPlatform(url) {
  // strip url first. Only interested in the platform part (assuming www.[platform].com)
  url = url.replace('http://', '');
  url = url.replace('https://', '');
  url = url.replace('www.', '');
  return url.split('.')[0];
}

// get currentTab
getCurrentTab().then((tab) => {
  chrome.runtime.sendMessage({action: "checkDuplicate", title: tab.title}, (response) => {
    if (!response) {
      activeTabUrl = tab.url;
      activeTabTitle = tab.title;
      urlField.value = activeTabUrl;
      titleField.value = activeTabTitle;
      platformField.value = getPlatform(activeTabUrl);
      // load data from datum
      chrome.storage.local.get('datum', function(result) {
        if (result.datum) {
          document.getElementById('singer').value = result.datum.singer;
          document.getElementById('model').value = result.datum.model;
          document.getElementById('language').value = result.datum.language;
          if (activeTabTitle.includes("AI")) {
            document.getElementById('bonafide_or_deepfake').checked = true;
          } else if (result.datum.bonafide_or_spoof == "deepfake") {
            document.getElementById('bonafide_or_deepfake').checked = true;
          } else {
            document.getElementById('bonafide_or_deepfake').checked = false;
          }
        }
      });
    } else {
      // replace entire body
      mainField.innerHTML = "<h5>Entry already exists!</h5>";
    }
  });
});

function bonafide_or_deepfake(unformatted) {
  if (unformatted) {
    return "deepfake";
  } else {
    return "bonafide";
  }
}

function handleButtonClick() {
  // Get values from field.
  datum = {
    url: urlField.value,
    title: titleField.value,
    platform: platformField.value,
    singer: document.getElementById('singer').value,
    model: document.getElementById('model').value,
    language: document.getElementById('language').value,
    bonafide_or_deepfake: bonafide_or_deepfake(document.getElementById('bonafide_or_deepfake').checked),
    submission_time: Date.now()
  };

  // store the datum in localStorage
  chrome.storage.local.set({ datum: datum });

  chrome.runtime.sendMessage({ action: "addData", data: datum }, (response) => {
    if (response.id) {
      console.log("Data added with ID:", response.id);
      alert("Data added with ID: " + response.id);
      // close current tab.
      // replace entire body
      mainField.innerHTML = "<h5>Entry already exists!</h5>";
    } else if (response.error) {
      console.error("Error adding data:", response.error);
      alert("Error adding data: " + response.error);
    }
  });
}