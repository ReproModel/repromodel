import React from 'react';

const MobileWarning = () => {
  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Oops!</h1>
      <p style={styles.message}>Repromodel is only built for Desktop use.</p>
      <p style={styles.message}>Looks like you're trying to access it on a mobile device. Trust us, it's way better on a big screen!</p>
      <p style={styles.footer}>Please visit us from a desktop for the full experience.</p>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    textAlign: 'center',
    padding: '20px',
    backgroundColor: '#f8f9fa',
  },
  header: {
    fontSize: '2.5em',
    margin: '0.5em',
  },
  message: {
    fontSize: '1.5em',
    margin: '0.5em',
  },
  footer: {
    fontSize: '1.2em',
    marginTop: '2em',
    color: '#6c757d',
  },
};

export default MobileWarning;