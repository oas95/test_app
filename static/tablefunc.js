function populateTable() {
  // Get the table bodies
  var expectedTableBody = document.getElementById('tableBody');
  var longlistTableBody = document.getElementById('longlist');
  var scoreTableBody = document.getElementById('score');
  var percentTableBody = document.getElementById('percentBody');

  // Clear any existing rows
  expectedTableBody.innerHTML = '';
  longlistTableBody.innerHTML = '';
  scoreTableBody.innerHTML = '';
  percentTableBody.innerHTML = '';

  // Get the rider and bull values from the form inputs
  const riders = [];
  for (let i = 1; i <= 7; i++) {
    riders.push(document.getElementById(`rider${i}`).value);
  }

  const bulls = [];
  for (let i = 1; i <= 5; i++) {
    bulls.push(document.getElementById(`bull${i}`).value);
  }

  // Create the data object
  const data = {
    rider1: riders[0],
    rider2: riders[1],
    rider3: riders[2],
    rider4: riders[3],
    rider5: riders[4],
    rider6: riders[5],
    rider7: riders[6],
    bull1: bulls[0],
    bull2: bulls[1],
    bull3: bulls[2],
    bull4: bulls[3],
    bull5: bulls[4]
  };

  // Send a POST request to the "/predict" endpoint
  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(data => {
      // data is the array of optimal assignments returned by the server
      for (var i = 0; i < data.length; i++) {
        // Create a new row and cells
        var row = document.createElement('tr');
        var cellRider = document.createElement('td');
        var cellBull = document.createElement('td');
        var cellProbability = document.createElement('td');

        // Set cell text
        cellRider.innerText = data[i].Rider;
        cellBull.innerText = data[i].Bull;
        cellProbability.innerText = data[i]['Success Probability'];

        // Append the cells to the row
        row.appendChild(cellRider);
        row.appendChild(cellBull);
        row.appendChild(cellProbability);

        // Append the row to the expected score lineup table body
        expectedTableBody.appendChild(row);
      }
    });

  // Send a POST request to the "/merged" endpoint
  fetch('/merged', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
    // data is the array of expected score assignments returned by the server
    for (var i = 0; i < data.length; i++) {
      // Create a new row and cells
      var row = document.createElement('tr');
      var cellRider = document.createElement('td');
      var cellBull = document.createElement('td');
      var cellProbability = document.createElement('td');
      var cellScore = document.createElement('td');

      // Set cell text
      cellRider.innerText = data[i].Rider;
      cellBull.innerText = data[i].Bull;
      cellProbability.innerText = data[i]['Successful_Probability'];
      cellScore.innerText = data[i]['Predicted Score'];

      // Append the cells to the row
      row.appendChild(cellRider);
      row.appendChild(cellBull);
      row.appendChild(cellProbability);
      row.appendChild(cellScore);

        // Append the row to the long list table body
        longlistTableBody.appendChild(row);
      }
    });

  // Send a POST request to the "/altline" endpoint
  fetch('/altline', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(data => {
    // data is the array of expected score assignments returned by the server
      for (var i = 0; i < data.length; i++) {
      // Create a new row and cells
        var row = document.createElement('tr');
        var cellRider = document.createElement('td');
        var cellBull = document.createElement('td');
        var cellProbability = document.createElement('td');
        var cellScore = document.createElement('td');

        // Set cell text
        cellRider.innerText = data[i].Rider;
        cellBull.innerText = data[i].Bull;
        cellProbability.innerText = data[i]['Successful_Probability'];
        cellScore.innerText = data[i]['Predicted Score'];

        // Append the cells to the row
        row.appendChild(cellRider);
        row.appendChild(cellBull);
        row.appendChild(cellProbability);
        row.appendChild(cellScore);

        // Append the row to the score table body
        scoreTableBody.appendChild(row);
      }
    });

  // Send a POST request to the "/expected" endpoint
  fetch('/combined', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(data => {
      // data is the array of expected score assignments returned by the server
      for (var i = 0; i < data.length; i++) {
        // Create a new row and cells
        var row = document.createElement('tr');
        var cellRider = document.createElement('td');
        var cellBull = document.createElement('td');
        var cellProbability = document.createElement('td');
        var cellScore = document.createElement('td');

        // Set cell text
        cellRider.innerText = data[i].Rider;
        cellBull.innerText = data[i].Bull;
        cellProbability.innerText = data[i]['Successful_Probability'];
        cellScore.innerText = data[i]['Predicted Score'];
        

        // Append the cells to the row
        row.appendChild(cellRider);
        row.appendChild(cellBull);
        row.appendChild(cellProbability);
        row.appendChild(cellScore);
        

        // Append the row to the expected score lineup table body
        percentTableBody.appendChild(row);
      }
    });
}

document.getElementById('runModelButton').addEventListener('click', populateTable);

