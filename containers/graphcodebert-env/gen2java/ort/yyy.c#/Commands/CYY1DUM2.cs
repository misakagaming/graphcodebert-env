namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: CYY1DUM2_DUMMY_SERVER_ABS        Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:40:13
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class CYY1DUM2 : ABBase
  {
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.IYY10321_IA Iyy10321Ia;
    GEN.ORT.YYY.IYY10321_OA Iyy10321Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020197_esc_flag;
    //       +->   CYY1DUM2_DUMMY_SERVER_ABS         01/09/2024  13:40
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  ****************************************************************
    //     1 !  Server Action Block(s)
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     2 !  !!!!!!!!!!!!!!!!
    //     2 !  Public olmayan ve sadece server psteplerden kullanılan 
    //     2 !  her AB yi USE edin, view tanımı ve matching yapmayın.
    //     2 !  
    //     3 !  USE iyy10321_type_read_s
    //     4 *  USE cyyy9001_exception_hndlng_dflt
    //     5 *  USE cyy1a131_server_init
    //     6 *  USE cyyy9141_context_string_set
    //     7 *  USE cyy1a121_server_termination
    //     8 *  USE cyyy9821_mv_ro1_to_yy1
    //     9 *  USE cyyy9831_mv_sc1_to_yy1
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public CYY1DUM2(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:40:13";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "CYY1DUM2_DUMMY_SERVER_ABS";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      _Execute();
    }
    
    private void _Execute()
    {
      
      ++(NestingLevel);
      try {
        f_22020197_init(  );
        f_22020197(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020197(  )
    {
      func_0022020197_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020197" );
      Globdata.GetStateData().SetCurrentABName( "CYY1DUM2_DUMMY_SERVER_ABS" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    Server Action Block(s)                                          
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!!!!!                                                
      //    Public olmayan ve sadece server psteplerden kullanılan          
      //    her AB yi USE edin, view tanımı ve matching yapmayın.           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000003" );
      
      Iyy10321Ia = (GEN.ORT.YYY.IYY10321_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.IYY10321).Assembly,
      	"GEN.ORT.YYY.IYY10321_IA" ));
      Iyy10321Oa = (GEN.ORT.YYY.IYY10321_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.IYY10321).Assembly,
      	"GEN.ORT.YYY.IYY10321_OA" ));
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.IYY10321).Assembly,
      	"GEN.ORT.YYY.IYY10321",
      	"Execute",
      	Iyy10321Ia,
      	Iyy10321Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020197" );
      Globdata.GetStateData().SetCurrentABName( "CYY1DUM2_DUMMY_SERVER_ABS" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000003" );
      Iyy10321Ia.FreeInstance(  );
      Iyy10321Ia = null;
      Iyy10321Oa.FreeInstance(  );
      Iyy10321Oa = null;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020197_init(  )
    {
      if ( NestingLevel < 2 )
      {
      }
    }
  }// end class
  
}
