$(function() {
    var riderInputs = ['#rider1', '#rider2', '#rider3', '#rider4', '#rider5', '#rider6', '#rider7'];
    var bullInputs = ['#bull1', '#bull2', '#bull3', '#bull4', '#bull5'];

    $.ajax({
        url: '/riders',  // Flask endpoint 
        type: 'GET',
        success: function(res) {
            riderInputs.forEach(function(input) {
                $(input).autocomplete({
                    source: function(request, response) {
                        var results = $.ui.autocomplete.filter(res, request.term).slice(0, 5);
                        response(results);
                    }
                });
            });
        },
        error: function(error) {
            console.log(error);
        }
    });

    $.ajax({
        url: '/bulls',  // Flask endpoint 
        type: 'GET',
        success: function(res) {
            bullInputs.forEach(function(input) {
                $(input).autocomplete({
                    source: function(request, response) {
                        var results = $.ui.autocomplete.filter(res, request.term).slice(0, 5);
                        response(results);
                    }
                });
            });
        },
        error: function(error) {
            console.log(error);
        }
    });
});

