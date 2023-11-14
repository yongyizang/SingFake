import { initializeApp } from "./firebase/firebase-app.js";
import { getFirestore, collection, query, where, getDocs, addDoc } from "./firebase/firebase-firestore.js";

const firebaseConfig = {
  apiKey: null,
  authDomain: null,
  projectId: null,
  storageBucket: null,
  messagingSenderId: null,
  appId: null,
  measurementId: null,
};

const firebase_app = initializeApp(firebaseConfig);
const db = getFirestore(firebase_app);

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'fetchData') {
    fetchData()
      .then(data => sendResponse({ data }))
      .catch(error => sendResponse({ error: error.message }));
    return true; 
  } else if (request.action === 'addData') {
    addData(request.data)
      .then(docRef => sendResponse({ id: docRef.id }))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  } else if (request.action == 'checkDuplicate') {
    checkDuplicateTitle(request.title)
    .then(count => sendResponse(count))
    .catch(error => sendResponse({ error: error.message }));
    return true; 
  }
});

// Add new data to Firestore
const addData = async (data) => {
  try {
    const docRef = await addDoc(collection(db, 'entries'), data);
    return docRef;
  } catch (error) {
    console.error('Error adding data:', error);
    throw error;
  }
};

// Fetch data from Firestore
const fetchData = async () => {
  try {
    const q = query(collection(db, "message"));
    const querySnapshot = await getDocs(q);
    const messages = querySnapshot.docs.map(doc => doc.data());
    return { messages };
  } catch (error) {
    console.error('Error getting documents:', error);
    throw error;
  }
};

const checkDuplicateTitle = async (title) => {
  try {
    const q = query(collection(db, "entries"), where("title", "==", title));
    const querySnapshot = await getDocs(q);
    // get length of querySnapshot
    const count = querySnapshot.docs.length;
    if (count > 0) {
      return true;
    } else {
      return false;
    }
  } catch (error) {
    console.error('Error getting length:', error);
    throw error;
  }
};