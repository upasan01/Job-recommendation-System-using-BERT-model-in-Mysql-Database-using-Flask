
$(document).ready(function() {

    $('#login_submit').on('click', function(){
    
    var login_form = $('#loginform').serialize();
    $.ajax({
       type: 'POST',
       url:'/login',
       data:login_form,
       success: function(response){
        
        console.log(response)
        window.alert(response)

       }
    });
    
    
    
    });
    });