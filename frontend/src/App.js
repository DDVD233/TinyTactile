import { useState, useEffect, useRef } from 'react';

function App() {
  
  const inputRef= useRef(null);

  const [text, setText] = useState("Init text");
  const [userInput, setUserInput] = useState("");

  useEffect(() => {
    inputRef.current.focus();
  }, []);

  return (
    <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          marginTop: "10em"
        }} 
    >
      <div
        style={{
          width: "60%",
          fontSize: "2rem"
        }}
      >
        {text}
      </div>
      <input
        type='text'
        ref={inputRef}
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        style={{
          width: "60%",
          marginTop: "2.5em",
          borderRadius: "0.5rem",
          border: "0.15rem solid black",
          fontSize: "2rem"
        }}
      />
    </div>
  );
}

export default App;
